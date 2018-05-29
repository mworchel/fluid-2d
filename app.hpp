#pragma once

#include "simulation.hpp"

#include <SFML/Graphics/RenderWindow.hpp>

#include <chrono>
#include <memory>

/**
 *
 */
class app {
public:
    app(int arg_count, char** args);

    void run();

private:
    struct time_state {
        std::chrono::milliseconds                          simulation_dt;
        std::chrono::time_point<std::chrono::steady_clock> start_time;
        std::chrono::nanoseconds                           elapsed_accumulated;
    };

    struct input_state {
        float movement_direction_x{ 0.f };
        float movement_direction_y{ 0.f };
        float manipulation_point_x{ 0.f };
        float manipulation_point_y{ 0.f };
    };

    struct source_state {
        bool  add_density{ false };
        bool  add_velocity{ false };
    };

    struct draw_state {
        bool              draw_density{ true };
        bool              draw_velocity{ false };
        color_multipliers density_color_multipliers{ 200.f, 100.f, 50.f };
    };


    static simulation_config parse_simulation_config(int arg_count, char** args);

    void handle_event(sf::Event const& event);

    void update();

    void draw();

    sf::RenderWindow m_window;
    time_state       m_time_state;
    input_state      m_input_state;
    source_state     m_source_state;
    draw_state       m_draw_state;
    simulation       m_simulation;
};
