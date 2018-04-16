#include "app.hpp"

using namespace std::chrono_literals;

app::app(int arg_count, char ** args)
  : m_window{ sf::VideoMode{ 640, 480 }, "Simulation" }
  , m_simulation{ app::parse_simulation_config(arg_count, args) }
{
  m_time_state.simulation_dt = 16ms;
}

void app::run()
{
  m_time_state.start_time = std::chrono::steady_clock::now();
  m_time_state.elapsed_accumulated = 0ns;

  while(m_window.isOpen())
  {
    sf::Event e;
    while(m_window.pollEvent(e))
    {
      handle_event(e);
    }

    update();

    draw();
  }
}

simulation_config app::parse_simulation_config(int arg_count, char ** args)
{
  simulation_config config;
  config.width = 1000U;
  config.height = 1000U;
  config.solver_type = solver_type::gpu; 
  return config;
}

void app::handle_event(sf::Event const & event)
{
  if(event.type == sf::Event::Closed)
  {
    m_window.close();
  }

  if(event.type == sf::Event::MouseButtonPressed)
  {
    m_input_state.manipulation_point_x = static_cast<float>(event.mouseButton.x);
    m_input_state.manipulation_point_y = static_cast<float>(event.mouseButton.y);

    if(event.mouseButton.button == sf::Mouse::Left)
    {
      m_input_state.add_density = true;
    }

    if(event.mouseButton.button == sf::Mouse::Right)
    {
      m_input_state.remove_density = true;
    }
  }

  if(event.type == sf::Event::MouseMoved)
  {
    m_input_state.manipulation_point_x = static_cast<float>(event.mouseMove.x);
    m_input_state.manipulation_point_y = static_cast<float>(event.mouseMove.y);
  }

  if(event.type == sf::Event::MouseButtonReleased)
  {
    if(event.mouseButton.button == sf::Mouse::Left)
    {
      m_input_state.add_density = false;
    }

    if(event.mouseButton.button == sf::Mouse::Right)
    {
      m_input_state.remove_density = false;
    }
  }
}

void app::update()
{
  // Perform simulation with the fixed simulation_dt timestep after at least simulation_dt time has passed
  auto finish_time = std::chrono::steady_clock::now();
  auto elapsed = finish_time - m_time_state.start_time;
  m_time_state.start_time = finish_time;
  m_time_state.elapsed_accumulated = m_time_state.elapsed_accumulated + elapsed;
  if(m_time_state.elapsed_accumulated >= m_time_state.simulation_dt)
  {
    m_time_state.elapsed_accumulated -= m_time_state.simulation_dt;

    // Add sources from input to the simulation
    if(m_input_state.add_density || m_input_state.remove_density)
    {
      size_t i;
      size_t j;
      if(m_simulation.to_density_cell(m_input_state.manipulation_point_x, m_input_state.manipulation_point_y, m_window, i, j))
      {
        float sign = m_input_state.add_density ? 1.f : -1.f;
        m_simulation.add_density_source(i, j, sign * 0.01f);
      }
    }

    m_simulation.update(m_time_state.simulation_dt);
  }
}

void app::draw()
{
  m_window.clear();
  m_simulation.draw(m_window);
  m_window.display();
}
