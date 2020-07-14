# A-reinforcement-learning-framework-for-multiple-comments-integration
 
[![GPL v3 licensed](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://github.com/98k-bot/GAN-Assisted-Preference-Based-Learning/blob/master/LICENSE) 

## Installation

An explicit spec file is not usually cross platform, and therefore has a comment at the top such as # platform: osx-64 showing the platform where it was created. This platform is the one where this spec file is known to work. On other platforms, the packages specified might not be available or dependencies might be missing for some of the key packages already in the spec.

To use the spec file to create an identical environment on the same machine or another machine:
```ruby
cd ~/venv
conda create --name rlenv --file spec-file.txt
```
To use the spec file to install its listed packages into an existing environment:
```ruby
conda install --name rlenv --file spec-file.txt
```
## Usage

### Train the agent:
```ruby
cd ~/Model
python agent.py
```
### Test the agent:
```ruby
cd ~/Model
python Eval.py
```
## License

_PPO_OIC_ is available under a GPLv3 license.

## Contributing

Fork and send a pull request. Or just e-mail us.
