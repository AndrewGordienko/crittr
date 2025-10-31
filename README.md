# crittr

**crittr** is a lightweight creature simulation library built with [Box2D](https://box2d.org/) and [pygame](https://www.pygame.org/).  
It provides simple abstractions for creating, simulating, and rendering articulated creatures in a 2D physics world.

## Features
- Physics simulation powered by Box2D
- Modular design for creatures, policies, and environments
- Built-in rendering with pygame
- Example scripts to get started quickly

## Installation
From PyPI (once published):
```bash
pip install crittr
```

For local development:
```bash
git clone https://github.com/yourusername/crittr.git
cd crittr
pip install -e .
```

## Quickstart
```python
from crittr import Simulator

sim = Simulator(CREATURE_NUMBER=10)

running = True
t = 0
while running:
    sim.step(t)
    sim.render()
    t += 1
```

Or run the demo:
```bash
python -m crittr.examples.demo
```

## Project Structure
```
crittr/
├── crittr/
│   ├── __init__.py
│   ├── physics.py
│   ├── creatures.py
│   ├── policies.py
│   ├── storage.py
│   └── examples/
│       └── demo.py
├── pyproject.toml
├── README.md
├── LICENSE
```

## Contributing
Pull requests are welcome! Please open an issue first to discuss any major changes.

## License
This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.
