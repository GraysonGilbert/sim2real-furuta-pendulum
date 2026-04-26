#include <SimpleFOC.h>
#include <SimpleFOCDrivers.h>
#include <encoders/esp32hwencoder/ESP32HWEncoder.h>
#include <SPI.h>
#include <AS5047P.h>
#include "driver/pcnt.h"
#include "policy_net.h" // Exported Trained Model


// BLDC Hardware Defintions:
ESP32HWEncoder bldc_encoder = ESP32HWEncoder(32, 33, 2048);
BLDCDriver3PWM driver = BLDCDriver3PWM(27, 26, 14, 12);
BLDCMotor motor = BLDCMotor(11);
InlineCurrentSense current_sense = InlineCurrentSense(0.01f, 50.0f, 34, 35);

// SPI Encoder Definition:
#define SLAVE_SELECT_PIN 5
AS5047P pend_encoder(SLAVE_SELECT_PIN, 10000000);
float pend_angle_offset = 0.0f;

// Low pass filter to reduce encoder noise
LowPassFilter lpf_pend_vel = LowPassFilter(0.005f);

// Safety Button Definition:
const int BUTTON_PIN = 13;

// Inter-core shared memory:
volatile float shared_target_current = 0.0f;
volatile bool is_estopped = false; // Temporarily stop motor
volatile bool hard_lockout = false; // Permanently stop motor
float prev_current_cmd = 0.0f; // Previous applied current

// Logging struct:
typedef struct {
  uint32_t timestamp;
  float obs[6];
  float action;
} LogFrame;

// Queue Handle
QueueHandle_t logQueue;

// --- BACKGROUND LOGGING TASK (Runs on Core 0, Low Priority) ---
void loggingTask(void * pvParameters) {
  LogFrame frame;
  
  // Print a CSV header
  Serial.println("Time_ms,Obs0,Obs1,Obs2,Obs3,Obs4,Obs5,Action");

  for(;;) {
    // Wait for data to appear in the queue. 
    if ( xQueueReceive(logQueue, &frame, portMAX_DELAY) == pdPASS ) {
      
      // Print the frame as a CSV line. 
      Serial.printf("%lu,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n", 
                    frame.timestamp, 
                    frame.obs[0], frame.obs[1], frame.obs[2], 
                    frame.obs[3], frame.obs[4], frame.obs[5], 
                    frame.action);
    }
  }
}

// Helper function for angle wrapping between [-pi, pi]
float wrap_angle(float angle)
{
  float wrapped = fmod(angle + PI, 2.0f * PI);
  if( wrapped < 0 )
  {
    wrapped += 2.0f * PI;
  }
  return wrapped - PI;
}

void encoder_cal(AS5047P encoder)
{
  Serial.println("Running pendulum encoder calibration...");
  delay(1000);

  float sum_sin = 0.0f;
  float sum_cos = 0.0f;
  const int num_samples = 100;

  for( int i = 0; i < num_samples; i++)
  {
    float raw_rad = encoder.readAngleDegree() * (PI / 180.0f);
    sum_sin += sin(raw_rad);
    sum_cos += cos(raw_rad);

    delay(2);
  }

  float avg_raw_angle = atan2(sum_sin, sum_cos) * (180.0f / PI);
  if( avg_raw_angle < 0 )
  {
    avg_raw_angle += 360.0f;
  }

  pend_angle_offset = 180.0f + avg_raw_angle;
  Serial.printf("Calibration complete. Rest angle: %.2f deg, Offset applied %.2f\n", avg_raw_angle, pend_angle_offset);
}

// Core 0: RL and Safety Tasks:
void core0_task(void * pvParameters)
{

  TickType_t xLastWakeTime = xTaskGetTickCount();
  const TickType_t xFrequency = pdMS_TO_TICKS(10);

  // Safety Button Variables:
  uint8_t click_count = 0;
  uint32_t last_press_time = 0;
  bool last_button_state = HIGH; // Input pullup to high

  const uint32_t CLICK_WINDOW = 800;
  const uint32_t DEBOUNCE_DELAY = 50;

  // Velocity Tracking Variables:
  float prev_pend_pos = 0.0f;
  uint32_t prev_time_us = micros();

  for( ;; )
  {
    // -------------------- Safety Button Logic --------------------
    uint32_t now = millis();
    bool current_button_state = digitalRead(BUTTON_PIN);

    if( current_button_state == LOW && last_button_state == HIGH )
    {
      if( now - last_press_time > DEBOUNCE_DELAY )
      {
        if( now - last_press_time <=  CLICK_WINDOW )
        {
          click_count++;
        }
        else
        {
          click_count = 1; // Press too slow, reset click count
        }
        last_press_time = now; // Update timestamp
      }
      if( click_count >= 3 )
      {
        hard_lockout = true;
        Serial.println("SAFETY LOCK OUT ENABLED!");
      }
    }
    last_button_state = current_button_state;
    // ----------------- End of Safety Button Logic -----------------

    // Read Pendulum Position and Compute Velocity:
    float angle_deg = pend_encoder.readAngleDegree() + pend_angle_offset;
    uint16_t magnitude = pend_encoder.readMagnitude();

    float pend_pos = -(angle_deg * (PI / 180.0f));

    uint32_t now_us = micros();
    float dt = (now_us - prev_time_us) * 1e-6f;

    if( dt <= 0.0f ) // Divide by zero check
    {
      dt = 0.0001f;
    }

    float delta_angle = wrap_angle(pend_pos - prev_pend_pos);
    float raw_pend_vel = delta_angle / dt;
    float pend_vel = lpf_pend_vel(raw_pend_vel);

    prev_pend_pos = pend_pos;
    prev_time_us = now_us;

    // Encoder Magnitude Check:
    bool magnet_fault = (magnitude < 2000);
    if( magnet_fault )
    {
      Serial.println(">> FAULT: Magnet magnitude too weak!");
    }

    // Check Safety Button State
    if( current_button_state == LOW || hard_lockout )
    {
      is_estopped = true;
      shared_target_current = 0.0f;
    }
    else
    {
      is_estopped = false;
    }
   
    // Format Observations and Run RL Model:
    float rotor_pos = motor.shaft_angle;
    float rotor_vel = motor.shaft_velocity;
    float pend_pos_wrapped = wrap_angle(pend_pos);

    float obs[6];
    obs[0] = constrain(rotor_pos / (4.0f * PI), -1.0f, 1.0f);
    obs[1] = constrain(rotor_vel / 50.0f, -1.0f, 1.0f);
    obs[2] = sin(pend_pos_wrapped);
    obs[3] = cos(pend_pos_wrapped);
    obs[4] = constrain(pend_vel / 50.0f, -1.0f, 1.0f);
    obs[5] = prev_current_cmd / 1.5f;

    if( !is_estopped )
    {
      float raw_action[1];
      compute_action(obs, raw_action);

      float calculated_target = constrain(raw_action[0], -1.0f, 1.0f) * 1.5f;
      prev_current_cmd = calculated_target;
      shared_target_current = calculated_target;

      // // SHADOW MODE TEST:
      // shared_target_current = 0.0f;

      // Serial.print(pend_pos);
      // Serial.print(",");
      // Serial.print(pend_pos_wrapped);
      // Serial.print(",");
      // Serial.println(calculated_target);

      LogFrame current_frame;
      current_frame.timestamp = now_us;
      memcpy(current_frame.obs, obs, sizeof(obs));
      current_frame.action = calculated_target;

      xQueueSend(logQueue, &current_frame, 0);

    }
    else
    {
      shared_target_current = 0.0f;
      prev_current_cmd = 0.0f;
    }
 
    vTaskDelayUntil(&xLastWakeTime, xFrequency); // 100Hz
  }
}


void setup()
{
  Serial.begin(921600); 

  // Create a queue capable of holding 20 frames of data
  logQueue = xQueueCreate(20, sizeof(LogFrame));

  // Launch the Logging Task on Core 0, but with Priority 0 (Lowest)
  // Your BrainTask is Priority 1, so it will always interrupt the logger.
  xTaskCreatePinnedToCore(
    loggingTask,   
    "LogTask",     
    4000,          // Stack size 
    NULL,          
    0,             // Priority 0 (Background)
    NULL,          
    0              // Core 0
  );

  delay(1000); 

  // Initialize SPI encoder
  pend_encoder.initSPI();
  Serial.println("Pendulum Encoder Initialized.");
  encoder_cal(pend_encoder);

  // Initialize Button Pin
  pinMode(BUTTON_PIN, INPUT_PULLUP);
  Serial.println("Safety Button Initialized.");

  // Initialize BLDC encoder and Driver
  bldc_encoder.init();
  motor.linkSensor(&bldc_encoder);

  pcnt_counter_pause(PCNT_UNIT_0);
  pcnt_counter_clear(PCNT_UNIT_0);
  pcnt_event_enable(PCNT_UNIT_0, PCNT_EVT_H_LIM);
  pcnt_counter_resume(PCNT_UNIT_0);

  driver.voltage_power_supply = 20;
  driver.init();
  motor.linkDriver(&driver);

  // Controller Configuration
  motor.phase_resistance = 5.55;
  motor.current_limit = 1.5;
  motor.voltage_limit = 20;   
  motor.velocity_index_search = 3;

  // Current Sense Link
  current_sense.linkDriver(&driver);
  current_sense.init();
  motor.linkCurrentSense(&current_sense);

  // FOC Current Settings
  motor.torque_controller = TorqueControlType::foc_current; 
  motor.controller = MotionControlType::torque;

  // Initialize & Align
  motor.init();
  motor.initFOC(); 
  Serial.println("Motor Ready!");

  // Create Core 0 Task
  xTaskCreatePinnedToCore(
    core0_task,    // Function to implment task
    "BrainTask",  // Name of task
    10000,        // Stack size of task
    NULL,         // Task input parameter
    1,            // Priority of the task
    NULL,         // Task handle
    0             // Core where task runs
  );

  Serial.println("Dual-Core RTOS Started");
}

void loop()
{
  bldc_encoder.update();
  motor.loopFOC();

  // If estopped apply 0 torque
  if( is_estopped )
  {
    motor.target = 0.0f;
  }
  else
  {
    motor.target = shared_target_current;
  }
  motor.move();
}