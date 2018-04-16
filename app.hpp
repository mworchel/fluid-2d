#pragma once

#include "simulation.hpp"

#include <SFML/Graphics/RenderWindow.hpp>

#include <chrono>
#include <memory>

class app
{
public:
  app(int arg_count, char** args);

  void run();

private:
  struct input_state
  {
    bool  add_density = false;
    bool  remove_density = false;
    float manipulation_point_x = 0.f;
    float manipulation_point_y = 0.f;
  };

  struct time_state
  {
    std::chrono::milliseconds                          simulation_dt;
    std::chrono::time_point<std::chrono::steady_clock> start_time;
    std::chrono::nanoseconds                           elapsed_accumulated;
  };

  static simulation_config parse_simulation_config(int arg_count, char** args);

  void handle_event(sf::Event const& event);

  void update();

  void draw();

  sf::RenderWindow m_window;
  time_state       m_time_state;
  input_state      m_input_state;
  simulation       m_simulation;
};
