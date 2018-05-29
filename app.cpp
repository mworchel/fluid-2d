#include "app.hpp"

using namespace std::chrono_literals;

app::app(int arg_count, char ** args)
    : m_window{ sf::VideoMode{ 800, 800 }, "Simulation" }
    , m_simulation{ parse_simulation_config(arg_count, args) } {
    m_time_state.simulation_dt = 20ms;
}

void app::run() {
    m_time_state.start_time = std::chrono::steady_clock::now();
    m_time_state.elapsed_accumulated = 0ns;

    while(m_window.isOpen()) {
        sf::Event e;
        while(m_window.pollEvent(e)) {
            handle_event(e);
        }

        update();

        draw();
    }
}

simulation_config app::parse_simulation_config(int arg_count, char ** args) {
    // TODO: Parse the command line options
    simulation_config config;
    config.width = 800U;
    config.height = 800U;
    config.solver_type = solver_type::gpu;
    config.diffusion_rate = 0.5f;
    config.viscosity = 0.000001f;
    return config;
}

void app::handle_event(sf::Event const & event) {
    if(event.type == sf::Event::Closed) {
        m_window.close();
    }

    if(event.type == sf::Event::KeyReleased) {
        if(event.key.code == sf::Keyboard::R) {
            m_simulation.reset();
        }

        if(event.key.code == sf::Keyboard::V) {
            m_draw_state.draw_velocity = !m_draw_state.draw_velocity;
        }

        if(event.key.code == sf::Keyboard::D) {
            m_draw_state.draw_density = !m_draw_state.draw_density;
        }
    }

    if(event.type == sf::Event::MouseButtonPressed) {
        m_input_state.manipulation_point_x = static_cast<float>(event.mouseButton.x);
        m_input_state.manipulation_point_y = static_cast<float>(event.mouseButton.y);
        m_input_state.movement_direction_x = 0.f;
        m_input_state.movement_direction_y = 0.f;

        if(event.mouseButton.button == sf::Mouse::Left) {
            m_source_state.add_density = true;
        }

        if(event.mouseButton.button == sf::Mouse::Right) {
            m_source_state.add_velocity = true;
        }
    }

    if(event.type == sf::Event::MouseMoved) {
        float current_x = static_cast<float>(event.mouseMove.x);
        float current_y = static_cast<float>(event.mouseMove.y);
        m_input_state.movement_direction_x = current_x - m_input_state.manipulation_point_x;
        m_input_state.movement_direction_y = current_y - m_input_state.manipulation_point_y;
        m_input_state.manipulation_point_x = current_x;
        m_input_state.manipulation_point_y = current_y;
    }

    if(event.type == sf::Event::MouseButtonReleased) {
        if(event.mouseButton.button == sf::Mouse::Left) {
            m_source_state.add_density = false;
        }

        if(event.mouseButton.button == sf::Mouse::Right) {
            m_source_state.add_velocity = false;
        }
    }
}

void app::update() {
    // Perform simulation with the fixed simulation_dt timestep after at least simulation_dt time has passed
    auto current_time = std::chrono::steady_clock::now();
    auto elapsed = current_time - m_time_state.start_time;
    m_time_state.start_time = current_time;
    m_time_state.elapsed_accumulated = m_time_state.elapsed_accumulated + elapsed;
    if(m_time_state.elapsed_accumulated >= m_time_state.simulation_dt) {
        m_time_state.elapsed_accumulated -= m_time_state.simulation_dt;

        // Add density to the position of the current manipulation point
        if(m_source_state.add_density) {
            size_t i;
            size_t j;
            if(m_simulation.to_density_cell(m_input_state.manipulation_point_x, m_input_state.manipulation_point_y, m_window, i, j)) {
                m_simulation.add_density_source(i, j, 0.075f);
            }
        }

        // Add velocity to the position of the current manipulation point in direction of the input movement
        if(m_source_state.add_velocity) {
            size_t i;
            size_t j;
            if(m_simulation.to_velocity_cell(m_input_state.manipulation_point_x, m_input_state.manipulation_point_y, m_window, i, j)) {
                m_simulation.add_velocity_source(i, j,
                                                 0.05f * m_input_state.movement_direction_x,
                                                 0.05f * m_input_state.movement_direction_y);
            }
        }

        m_simulation.update(m_time_state.simulation_dt);
    }
}

void app::draw() {
    m_window.clear();
    m_simulation.draw(m_window, m_draw_state.density_color_multipliers, m_draw_state.draw_density, m_draw_state.draw_velocity);
    m_window.display();
}
