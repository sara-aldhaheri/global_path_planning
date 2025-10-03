// Copyright (c) 2025 Technology Innovation Institute. All Rights Reserved.

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <nav_msgs/msg/path.hpp>
#include <std_srvs/srv/trigger.hpp>

#include <vector>
#include <array>
#include <cmath>
#include <memory>
#include <string>
#include <iostream>
#include <mutex>

// GPMP2 and GTSAM includes
#include <gtsam/geometry/Point2.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/DoglegOptimizer.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>
#include <gtsam/linear/NoiseModel.h>

#include <gpmp2/planner/BatchTrajOptimizer.h>
#include <gpmp2/kinematics/PointRobot.h>
#include <gpmp2/kinematics/PointRobotModel.h>
#include <gpmp2/obstacle/PlanarSDF.h>
#include <gpmp2/obstacle/ObstaclePlanarSDFFactorPointRobot.h>
#include <gpmp2/obstacle/ObstaclePlanarSDFFactorGPPointRobot.h>
#include <gpmp2/gp/GaussianProcessPriorLinear.h>

using namespace gtsam;
using namespace gpmp2;


class GPMP2PlannerNode : public rclcpp::Node {
public:
    GPMP2PlannerNode() : Node("gpmp2_planner_cpp") {
        // Declare parameters with default values
        declare_parameter("sample_map.robot_radius", 3.0);
        declare_parameter("sample_map.map_width", 40.0);
        declare_parameter("sample_map.map_height", 40.0);
        declare_parameter("sample_map.map_resolution", 0.1);
        declare_parameter("sample_map.start_x", -6.0);
        declare_parameter("sample_map.start_y", -6.0);
        declare_parameter("sample_map.goal_x", 16.0);
        declare_parameter("sample_map.goal_y", 12.0);
        declare_parameter("sample_map.obstacle_radius", 2.0);
        declare_parameter("total_time_step", 10);
        declare_parameter("enable_auto_replanning", true);

        // Get parameters
        robot_radius_   = get_parameter("sample_map.robot_radius").as_double();
        map_width_      = get_parameter("sample_map.map_width").as_double();
        map_height_     = get_parameter("sample_map.map_height").as_double();
        map_resolution_ = get_parameter("sample_map.map_resolution").as_double();
        default_start_  ={get_parameter("sample_map.start_x").as_double(), 
                          get_parameter("sample_map.start_y").as_double()};
        default_goal_   = {get_parameter("sample_map.goal_x").as_double(), 
                          get_parameter("sample_map.goal_y").as_double()};
        obstacle_radius_= get_parameter("sample_map.obstacle_radius").as_double();
        total_time_step = get_parameter("total_time_step").as_int();
        enable_auto_replanning_ = get_parameter("enable_auto_replanning").as_bool();
        
        // Initialize current goal with default
        // std::lock_guard<std::mutex> lock(goal_mutex_);
        current_goal_ = default_goal_;
        
        // Obstacles
        obstacle_centers_ = {
            {-5.0, -2.0}, {0.0, -5.0}, {5.0, 2.0}, 
            {-2.0, 5.0}, {2.0, -1.0}, {12.0, 8.0}, {-10.0, 10.0}
        };
        obstacle_radius_ = 2.0;
        
        // ROS setup
        plan_service_ = this->create_service<std_srvs::srv::Trigger>(
            "plan_path", 
            std::bind(&GPMP2PlannerNode::plan_path_callback, this, 
                      std::placeholders::_1, std::placeholders::_2)
        );
        
        // Subscriber for 2D Nav Goal from RViz
        goal_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            "/goal_pose", 10,
            std::bind(&GPMP2PlannerNode::goal_pose_callback, this, std::placeholders::_1)
        );
        
        // Publishers
        path_pub_ = this->create_publisher<nav_msgs::msg::Path>("planned_path", 10);
        map_pub_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>("occupancy_grid", 10);
        initial_pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("/initialpose", 10);
        goal_marker_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("/move_base_simple/goal", 10);
        
        // Initialize
        initialize_map();
        
        // Execute every 1 second
        path_timer_ = this->create_wall_timer(
            std::chrono::seconds(1),
            std::bind(&GPMP2PlannerNode::publish_path_once, this)
        );
        
        map_timer_ = this->create_wall_timer(
            std::chrono::seconds(5),
            std::bind(&GPMP2PlannerNode::publish_map, this)
        );
        
        RCLCPP_INFO(this->get_logger(), "GPMP2 Planner initialized");
        RCLCPP_INFO(this->get_logger(), "Auto-replanning: %s", 
                   enable_auto_replanning_ ? "ENABLED" : "DISABLED");
        RCLCPP_INFO(this->get_logger(), "Use '2D Nav Goal' in RViz to set new goal poses");
    }

private:
    // Parameters
    double robot_radius_;
    double map_width_, map_height_, map_resolution_;
    std::array<double, 2> default_start_, default_goal_;
    std::array<double, 2> current_goal_;
    std::vector<std::array<double, 2>> obstacle_centers_;
    double obstacle_radius_;
    int total_time_step;
    bool enable_auto_replanning_;
    
    // Thread safety
    std::mutex goal_mutex_;
    
    // ROS components
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr plan_service_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr goal_sub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr map_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr initial_pose_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr goal_marker_pub_;
    rclcpp::TimerBase::SharedPtr path_timer_, map_timer_;
    
    // Map data
    nav_msgs::msg::OccupancyGrid occupancy_grid_;
    
    std::shared_ptr<PlanarSDF> sdf_map_;

    void goal_pose_callback(const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
        // Validate the goal is within map bounds
        double goal_x = msg->pose.position.x;
        double goal_y = msg->pose.position.y;
        
        double map_min_x = -map_width_ / 2;
        double map_max_x = map_width_ / 2;
        double map_min_y = -map_height_ / 2;
        double map_max_y = map_height_ / 2;
        
        if (goal_x < map_min_x || goal_x > map_max_x || 
            goal_y < map_min_y || goal_y > map_max_y) {
            RCLCPP_WARN(this->get_logger(), 
                       "Goal pose (%.2f, %.2f) is outside map bounds [%.2f, %.2f] x [%.2f, %.2f]. Ignoring.",
                       goal_x, goal_y, map_min_x, map_max_x, map_min_y, map_max_y);
            return;
        }
        
        // Check if goal is too close to obstacles
        bool goal_valid = true;
        double safety_margin = robot_radius_ + 0.5; // Add some safety margin. TODO: make configurable or less abstract
        
        for (const auto& obstacle : obstacle_centers_) {
            double dist_to_obstacle = std::sqrt(
                std::pow(goal_x - obstacle[0], 2) + 
                std::pow(goal_y - obstacle[1], 2)
            );
            
            if (dist_to_obstacle < (obstacle_radius_ + safety_margin)) {
                RCLCPP_WARN(this->get_logger(), 
                           "Goal pose (%.2f, %.2f) is too close to obstacle at (%.2f, %.2f). Distance: %.2f, Required: %.2f",
                           goal_x, goal_y, obstacle[0], obstacle[1], dist_to_obstacle, obstacle_radius_ + safety_margin);
                goal_valid = false;
                break;
            }
        }
        
        if (!goal_valid) {
            RCLCPP_WARN(this->get_logger(), "Goal pose rejected due to obstacle collision");
            return;
        }
        
        // Update current goal
        {
            std::lock_guard<std::mutex> lock(goal_mutex_);
            current_goal_[0] = goal_x;
            current_goal_[1] = goal_y;
        }
        
        RCLCPP_INFO(this->get_logger(), "New goal received: (%.3f, %.3f)", goal_x, goal_y);
        
        // Echo the goal back for visualization
        auto goal_marker = *msg;
        goal_marker.header.stamp = this->get_clock()->now();
        goal_marker_pub_->publish(goal_marker);
        
        // Trigger replanning if auto-replanning is enabled
        if (enable_auto_replanning_) {
            RCLCPP_INFO(this->get_logger(), "Auto-replanning triggered...");
            replan_path();
        } else {
            RCLCPP_INFO(this->get_logger(), "Goal updated.");
        }
    }
    
    void replan_path() {
        std::array<double, 2> goal_copy;
        {
            std::lock_guard<std::mutex> lock(goal_mutex_);
            goal_copy = current_goal_;
        }
        
        auto path_points = generate_path_gpmp2(default_start_, goal_copy);
        if (!path_points.empty()) {
            publish_path(path_points);
            RCLCPP_INFO(this->get_logger(), "Successfully replanned path to goal (%.3f, %.3f)", 
                       goal_copy[0], goal_copy[1]);
        } else {
            RCLCPP_ERROR(this->get_logger(), "Failed to replan path to goal (%.3f, %.3f)", 
                        goal_copy[0], goal_copy[1]);
        }
    }

    void initialize_map() {
        int width_cells = static_cast<int>(map_width_ / map_resolution_);
        int height_cells = static_cast<int>(map_height_ / map_resolution_);
        
        // Create occupancy grid
        occupancy_grid_.header.frame_id = "map";
        occupancy_grid_.info.resolution = map_resolution_;
        occupancy_grid_.info.width = width_cells;
        occupancy_grid_.info.height = height_cells;
        occupancy_grid_.info.origin.position.x = -map_width_ / 2;
        occupancy_grid_.info.origin.position.y = -map_height_ / 2;
        occupancy_grid_.info.origin.orientation.w = 1.0;
        occupancy_grid_.data.resize(width_cells * height_cells, 0);
        
        // Add obstacles to occupancy grid
        for (const auto& center : obstacle_centers_) {
            double center_x = center[0];
            double center_y = center[1];
            
            for (int x = 0; x < width_cells; ++x) {
                for (int y = 0; y < height_cells; ++y) {
                    double world_x = (x * map_resolution_) - (map_width_ / 2);
                    double world_y = (y * map_resolution_) - (map_height_ / 2);
                    
                    double dist = std::sqrt(std::pow(world_x - center_x, 2) + 
                                           std::pow(world_y - center_y, 2));
                    
                    if (dist <= obstacle_radius_) {
                        int index = y * width_cells + x;
                        occupancy_grid_.data[index] = 100;
                    }
                }
            }
        }
        
        // Create SDF for GPMP2
        create_sdf_map(width_cells, height_cells);
    }

void create_sdf_map(int width_cells, int height_cells) {
    try {
        // Create field data matrix
        Matrix field_data(height_cells, width_cells);
        
        for (int y = 0; y < height_cells; ++y) {
            for (int x = 0; x < width_cells; ++x) {
                double world_x = (x * map_resolution_) - (map_width_ / 2);
                double world_y = (y * map_resolution_) - (map_height_ / 2);
                
                // Find minimum distance to any obstacle
                double min_distance = std::numeric_limits<double>::max();
                for (const auto& center : obstacle_centers_) {
                    double dist_to_center = std::sqrt(
                        std::pow(world_x - center[0], 2) + 
                        std::pow(world_y - center[1], 2)
                    );
                    min_distance = std::min(min_distance, dist_to_center);
                }
                
                // Signed distance (negative inside obstacles)
                double signed_distance = min_distance - obstacle_radius_;
                field_data(y, x) = signed_distance;
            }
        }
        
        // Create SDF
        Point2 origin(-map_width_ / 2, -map_height_ / 2);
        sdf_map_ = std::make_shared<PlanarSDF>(origin, map_resolution_, field_data);
        
        RCLCPP_INFO(this->get_logger(), "SDF created successfully");
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "SDF creation failed: %s", e.what());
        sdf_map_ = nullptr;
    }
}

std::vector<std::array<double, 2>> generate_path_gpmp2(
    const std::array<double, 2>& start, 
    const std::array<double, 2>& goal) {
    
    if (!sdf_map_) {
        RCLCPP_ERROR(this->get_logger(), "SDF map not available");
        return {};
    }
    
    try {
        // Configuration space (x, y, theta). TODO: include theta planning
        Vector start_conf(3);
        start_conf << start[0], start[1], -1.0;
        Vector end_conf(3);
        end_conf << goal[0], goal[1], 0.0;
        Vector start_vel = Vector::Zero(3);
        Vector end_vel = Vector::Zero(3);
        Vector start_acc = Vector::Zero(3);
        Vector end_acc = Vector::Zero(3);
        
        // GPMP2 parameters
        double total_time_sec = 2.5;
        total_time_step = this->get_parameter("total_time_step").as_int();
        double delta_t = total_time_sec / total_time_step;
        
        // Robot model
        PointRobot pR(3, 1);
        BodySphereVector sphere_vec;
        Matrix center = Matrix::Zero(3, 1);
        sphere_vec.push_back(BodySphere(0, robot_radius_, center));
        PointRobotModel pR_model(pR, sphere_vec);
        
        // Noise models
        Matrix Qc_pos = Matrix::Identity(pR_model.dof(), pR_model.dof());
        Matrix Qc_vel = Matrix::Identity(pR_model.dof(), pR_model.dof()) * 0.1;
        auto Qc_pos_model = noiseModel::Gaussian::Covariance(Qc_pos);
        auto Qc_vel_model = noiseModel::Gaussian::Covariance(Qc_vel);
        
        auto pose_fix = noiseModel::Isotropic::Sigma(pR_model.dof(), 0.0001);
        auto vel_fix = noiseModel::Isotropic::Sigma(pR_model.dof(), 0.0001);
        auto acc_fix = noiseModel::Isotropic::Sigma(pR_model.dof(), 0.0001);
        
        // Initialize factor graph
        NonlinearFactorGraph graph;
        Values init_values;
        
        // Build trajectory
        for (int i = 0; i <= total_time_step; ++i) {
            Key key_pos = Symbol('x', i);
            Key key_vel = Symbol('v', i);
            Key key_acc = Symbol('a', i);
            
            // Proper interpolation between start and goal
            double ratio = static_cast<double>(i) / static_cast<double>(total_time_step);
            Vector pose = start_conf + ratio * (end_conf - start_conf);
            Vector vel = (end_conf - start_conf) / total_time_sec;
            Vector acc = (vel - start_vel) / total_time_sec;
            
            init_values.insert(key_pos, pose);
            init_values.insert(key_vel, vel);
            init_values.insert(key_acc, acc);
            
            // Start constraints
            if (i == 0) {
                graph.emplace_shared<PriorFactor<Vector>>(key_pos, start_conf, pose_fix);
                graph.emplace_shared<PriorFactor<Vector>>(key_vel, start_vel, vel_fix);
                graph.emplace_shared<PriorFactor<Vector>>(key_acc, start_acc, acc_fix);
            }
            // End constraints
            else if (i == total_time_step) {
                graph.emplace_shared<PriorFactor<Vector>>(key_pos, end_conf, pose_fix);
                graph.emplace_shared<PriorFactor<Vector>>(key_vel, end_vel, vel_fix);
                graph.emplace_shared<PriorFactor<Vector>>(key_acc, end_acc, acc_fix);
            }
            
            // GP priors and obstacle avoidance
            if (i > 0) {
                Key key_pos1 = Symbol('x', i - 1);
                Key key_pos2 = Symbol('x', i);
                Key key_vel1 = Symbol('v', i - 1);
                Key key_vel2 = Symbol('v', i);
                
                // GP prior for position-velocity chain
                graph.emplace_shared<GaussianProcessPriorLinear>(
                    key_pos1, key_vel1, key_pos2, key_vel2, delta_t, Qc_pos_model
                );
                
                // GP prior for velocity-acceleration chain
                if (i > 1) {
                    Key key_acc1 = Symbol('a', i - 1);
                    Key key_acc2 = Symbol('a', i);
                    
                    graph.emplace_shared<GaussianProcessPriorLinear>(
                        key_vel1, key_acc1, key_vel2, key_acc2, delta_t, Qc_vel_model
                    );
                }
                
                // Obstacle avoidance
                double cost_sigma = 0.05;
                double epsilon_dist = 4.0;
                graph.emplace_shared<ObstaclePlanarSDFFactorPointRobot>(
                    key_pos2, pR_model, *sdf_map_, cost_sigma, epsilon_dist
                );
                
                // Continuous collision checking
                for (int j = 1; j < 3; ++j) {
                    double tau = j * (delta_t / 3);
                    graph.emplace_shared<ObstaclePlanarSDFFactorGPPointRobot>(
                        key_pos1, key_vel1, key_pos2, key_vel2,
                        pR_model, *sdf_map_, cost_sigma, epsilon_dist,
                        Qc_pos_model, delta_t, tau
                    );
                }
            }
        }
        
        // Optimize
        DoglegParams parameters;
        parameters.setMaxIterations(100);
        DoglegOptimizer optimizer(graph, init_values, parameters);
        
        RCLCPP_INFO(this->get_logger(), "Initial Error: %f", graph.error(init_values));
        Values result = optimizer.optimize();
        RCLCPP_INFO(this->get_logger(), "Final Error: %f", graph.error(result));
        
        // Extract path and velocities with detailed output
        std::vector<std::array<double, 2>> path_points;
        
        RCLCPP_INFO(this->get_logger(), "=== PATH POINTS AND VELOCITIES ===");
        for (int i = 0; i <= total_time_step; ++i) {
            Key key_pos = Symbol('x', i);
            Key key_vel = Symbol('v', i);
            Key key_acc = Symbol('a', i);
            
            Vector pose = result.at<Vector>(key_pos);
            Vector velocity = result.at<Vector>(key_vel);
            Vector acceleration = result.at<Vector>(key_acc);
            
            path_points.push_back({pose[0], pose[1]});
            
            double speed = std::sqrt(velocity[0]*velocity[0] + velocity[1]*velocity[1]);
            
            RCLCPP_INFO(this->get_logger(),
                "Point %d: Position=(%.3f, %.3f), "
                "Velocity=(%.3f, %.3f), "
                "Acceleration=(%.3f, %.3f), "
                "Speed=%.3f",
                i, pose[0], pose[1], velocity[0], velocity[1], 
                acceleration[0], acceleration[1], speed
            );
        }
        
        // Calculate path statistics
        RCLCPP_INFO(this->get_logger(), "=== PATH SUMMARY ===");
        RCLCPP_INFO(this->get_logger(), "Total waypoints: %zu", path_points.size());
        RCLCPP_INFO(this->get_logger(), "Start: (%.3f, %.3f)", 
                    path_points[0][0], path_points[0][1]);
        RCLCPP_INFO(this->get_logger(), "Goal: (%.3f, %.3f)", 
                    path_points.back()[0], path_points.back()[1]);
        
        double path_length = 0.0;
        for (size_t i = 1; i < path_points.size(); ++i) {
            double dx = path_points[i][0] - path_points[i-1][0];
            double dy = path_points[i][1] - path_points[i-1][1];
            path_length += std::sqrt(dx*dx + dy*dy);
        }
        RCLCPP_INFO(this->get_logger(), "Path length: %.3f meters", path_length);
        
        RCLCPP_INFO(this->get_logger(), "GPMP2 generated %zu waypoints", path_points.size());
        return path_points;
        
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "GPMP2 failed: %s", e.what());
        return {};
    }
}


    void publish_map() {
        occupancy_grid_.header.stamp = this->get_clock()->now();
        map_pub_->publish(occupancy_grid_);
    }

    void publish_path_once() {
        std::array<double, 2> goal_copy;
        {
            std::lock_guard<std::mutex> lock(goal_mutex_);
            goal_copy = current_goal_;
        }
        
        auto path_points = generate_path_gpmp2(default_start_, goal_copy);
        if (!path_points.empty()) {
            publish_path(path_points);
        }
    }

    void publish_path(const std::vector<std::array<double, 2>>& path_points) {
        nav_msgs::msg::Path path_msg;
        path_msg.header.frame_id = "map";
        path_msg.header.stamp = this->get_clock()->now();
        
        for (const auto& point : path_points) {
            geometry_msgs::msg::PoseStamped pose;
            pose.header.frame_id = "map";
            pose.header.stamp = this->get_clock()->now();
            pose.pose.position.x = point[0];
            pose.pose.position.y = point[1];
            pose.pose.position.z = 0.0;
            pose.pose.orientation.w = 1.0;
            path_msg.poses.push_back(pose);
        }
        
        path_pub_->publish(path_msg);
        RCLCPP_INFO(this->get_logger(), "Published path with %zu poses", path_msg.poses.size());

        // Publish initial pose (first waypoint)
        if (!path_msg.poses.empty()) {
            initial_pose_pub_->publish(path_msg.poses[0]);
            RCLCPP_INFO(this->get_logger(), "Published initial pose at (%.3f, %.3f)", 
                       path_msg.poses[0].pose.position.x, 
                       path_msg.poses[0].pose.position.y);
        }
    }

    void plan_path_callback(
        const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
        std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
        
        (void)request; // Suppress unused parameter warning
        
        std::array<double, 2> goal_copy;
        {
            std::lock_guard<std::mutex> lock(goal_mutex_);
            goal_copy = current_goal_;
        }
        
        auto path_points = generate_path_gpmp2(default_start_, goal_copy);
        if (!path_points.empty()) {
            publish_path(path_points);
            response->success = true;
            response->message = "Path planned with " + std::to_string(path_points.size()) + 
                               " waypoints to goal (" + std::to_string(goal_copy[0]) + 
                               ", " + std::to_string(goal_copy[1]) + ")";
        } else {
            response->success = false;
            response->message = "Path planning failed for goal (" + std::to_string(goal_copy[0]) + 
                               ", " + std::to_string(goal_copy[1]) + ")";
        }
    }
};

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    
    auto node = std::make_shared<GPMP2PlannerNode>();
    
    try {
        rclcpp::spin(node);
    } catch (const std::exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("gpmp2_planner"), "Exception: %s", e.what());
    }
    
    rclcpp::shutdown();
    return 0;
}