#include <cmath> // for trigonometric functions
#include <iostream>
#include <random> // for random number generation

// Define the robot's starting position and velocity
bool atCheckPoint = false;
double current_x = 0.0;
double current_y = 0.0;
double current_theta = 0.0; // facing right (x-axis)
double target_v = 0.1; // meters per second
double original_target_v = target_v; // save the original target velocity

// Define the coordinates of the checkpoint and guard
double checkpoint_x = 3.0;
double checkpoint_y = 4.0;

// Define a function to calculate the distance between two points
double getDistance(double x1, double y1, double x2, double y2) {
    double dx = x2 - x1;
    double dy = y2 - y1;
    return sqrt(dx * dx + dy * dy);
}

// Define a function to update the position of the robot
void updatePosition(double delta_x, double delta_y) {
    current_x += delta_x;
    current_y += delta_y;
}

// Define a function to update the orientation of the robot
void updateOrientation(double delta_theta) {
    current_theta += delta_theta;
}

// Define a function to check the sound level and determine if the robot should stop
bool shouldStop(double sound_magnitude) {
    // Define the threshold sound level at which the robot should stop
    double max_sound_level = 0.5; // units of sound measurement

    // Define the maximum safe distance at which the robot will stop
    double max_safe_distance = 0.5; // meters

    // Check the sound level
    if (sound_magnitude > max_sound_level) {
        std::cout << "Guard detected. Stopping and backing up." << std::endl;
        target_v = -0.1; // back up at 0.1 m/s
        while (sound_magnitude > max_sound_level) {
            // continuously back up until safe distance is reached
            updatePosition(-target_v * cos(current_theta), -target_v * sin(current_theta));
            std::cout << "Backing up... " << std::endl;
        }
        std::cout << "Safe distance reached. Resuming original movement." << std::endl;
        target_v = original_target_v;
        return true; // robot has stopped
    }
    return false; // robot can continue moving
}

int main() {
    // Generate a random starting position for the robot within a square with side length 2
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-1.0, 1.0);
    current_x = dis(gen);
    current_y = dis(gen);

    // Turn towards the checkpoint
    double delta_x = checkpoint_x - current_x;
    double delta_y = checkpoint_y - current_y;
    double delta_theta = atan2(delta_y, delta_x) - current_theta;
    // Adjust the angle to be within -pi and pi
    if (delta_theta > M_PI) {
        delta_theta -= 2 * M_PI;
    } else if (delta_theta < -M_PI) {
        delta_theta += 2 * M_PI;
    }
        // Turn the robot towards the checkpoint
    double omega = 0.0;
    if (delta_theta > 0.0) {
        omega = 1.0; // turn left
    } else if (delta_theta < 0.0) {
        omega = -1.0; // turn right
    }
    while (std::abs(delta_theta) > 0.01) { // continue turning until close enough to the target orientation
        // calculate the new position and orientation
        double delta_t = 0.1; // seconds
        double v = 0.0;
        updateOrientation(omega * delta_t);
        updatePosition(v * cos(current_theta) * delta_t, v * sin(current_theta) * delta_t);
        // calculate the new orientation error
        delta_x = checkpoint_x - current_x;
        delta_y = checkpoint_y - current_y;
        delta_theta = atan2(delta_y, delta_x) - current_theta;
        // adjust the angle to be within -pi and pi
        if (delta_theta > M_PI) {
            delta_theta -= 2 * M_PI;
        } else if (delta_theta < -M_PI) {
            delta_theta += 2 * M_PI;
        }
        // adjust the turning direction
        if (delta_theta > 0.0) {
            omega = 1.0; // turn left
        } else if (delta_theta < 0.0) {
            omega = -1.0; // turn right
        }
    }

    std::cout << "Robot turned towards the checkpoint." << std::endl;

    // Drive towards the checkpoint
    while (getDistance(current_x, current_y, checkpoint_x, checkpoint_y) > 0.01) { // continue driving until close enough to the checkpoint
        // Check for the guard
        double sound_magnitude = 0.0; // read sound level from microphone
        shouldStop(sound_magnitude); // check if the robot should stop

        // calculate the new position and orientation
        double delta_t = 0.1; // seconds
        updatePosition(target_v * cos(current_theta) * delta_t, target_v * sin(current_theta) * delta_t);
        // calculate the new distance to the checkpoint
        delta_x = checkpoint_x - current_x;
        delta_y = checkpoint_y - current_y;

        std::cout << "Moving towards checkpoint... Current distance: " << getDistance(current_x, current_y, checkpoint_x, checkpoint_y) << std::endl;
    }

    std::cout << "Checkpoint reached. Robot stopped." << std::endl;
    atCheckPoint = true;

    return 0;
}
