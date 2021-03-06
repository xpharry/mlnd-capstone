# mlnd-capstone
MLND Part 7: Capstone Proposal & Capstone Project.

## Project: Capstone Proposal

- [Capstone Proposal](./proposal/proposal.pdf)

A project proposal is listed here and a game bot is expected to be built with Deep Reinforcement Learning method.

## Project: Capstone Project

Followed by the Capstone Proposal, a project is established just as indicated in the proposal.

**A Game Bot Trained with Deep Q-Learning**

- [Capstone Report](./report/report.pdf)

### Overview:

This is an attempt at Sirajology's OpenAI Universe [coding challenge](https://www.youtube.com/watch?v=mGYU5t8MO7s&t=11s).

Well, it is surely long after the coding challenge. It is still worthy to try since it gives me much intuitive into Deep Reinforcement Learning. Also I treat it as my capstone project in my Machine Learning Nano Degree Program. It could be a perfect wrap up since it combines both the Deep Learning and Reinforcement Learning and as a result it concludes a genereal pipeline to drive the computer (here it is the video game, Coaster Racer) to learn by itself. Just like the power it has shown in the famous competition between AlphaGo and Lee Sedo, Deep Reinforcement Learning has become a representative of Artificial Intelligence.

I personally feel motivated because I expect a great potential problem solving ability in my familiar areas, robotics and simulations. Video game bot is a perfect beginning and in the future the trained intelligence here can be transplanted to a real robot by Transfer Learning.

In this project, I start from simply a Q-Learning model and add in some deep learning models which has been proved powerful in other problems, like the model proposed in the DeepMind Nature paper, and also a model succefully trained a turtlebot moving in a maze. The model is expected to learn the flash game CoasterRacer based on its vision, namely the pixels of the screen, and the score generated by the game. Theoretically speaking the trained model would work for any implementation of a flash game within Universe providing the correct dimensions of the image crop and the similar inputs and outputs.

### Installation

#### [Install Universe](https://github.com/openai/universe#installation)

##### On Ubuntu 16.04:

```shell
pip install numpy
sudo apt-get install golang libjpeg-turbo8-dev make
```

##### On Ubuntu 14.04:

golang v1.9

```
$ sudo add-apt-repository ppa:gophers/archive
$ sudo apt-get update
$ sudo apt-get install golang-1.9-go
$ sudo apt-get install libjpeg-turbo8-dev make
```

**Note** that golang-1.9-go puts binaries in /usr/lib/go-1.9/bin.

If you want them on your PATH, you need to make that change yourself.

For example, 

```
sudo mv /usr/lib/go-1.9 /usr/local/go
```

Then, in your `~/.bashrc` file, add this line:

```
export PATH=$PATH:/usr/local/go/bin
```

##### On OSX:

You might need to install Command Line Tools by running:

```shell
xcode-select --install
```

Or ``numpy``, ``libjpeg-turbo`` and ``incremental`` packages:

```shell
pip install numpy incremental
brew install golang libjpeg-turbo
```

#### Install Docker

The majority of the environments in Universe run inside Docker
containers, so you will need to [install Docker](https://docs.docker.com/engine/installation/)(on OSX, we recommend [Docker for Mac](https://docs.docker.com/docker-for-mac/)). You should be able to run ``docker ps`` and get something like this:

```shell
 $ docker ps
 CONTAINER ID        IMAGE               COMMAND             CREATED             STATUS              PORTS               NAMES
```

Alternate configuration - running the agent in docker

The above instructions result in an agent that runs as a regular python process in your OS, and launches docker containers as needed for the remotes.

Alternatively, you can build a docker image for the agent and run it as a container as well.

You can do this in any operating system that has a recent version of docker installed, and the git client.

To get started, clone the ``universe`` repo:

```shell
git clone https://github.com/openai/universe.git
cd universe
```
	
Build a docker image, tag it as 'universe':

```shell
docker build -t universe .
```

This may take a while the first time, as the docker image layers are pulled from docker hub.

Once the image is built, you can do a quick run of the test cases to make sure everything is working:

```shell
docker run --privileged --rm -e DOCKER_NET_HOST=172.17.0.1 -v /var/run/docker.sock:/var/run/docker.sock universe pytest
```

Here's a breakdown of that command:

* ``docker run`` - launch a docker container
* ``--rm`` - delete the container once the launched process finishes
* ``-e DOCKER_NET_HOST=172.17.0.1`` - tells the universe remote (when launched) to make its VNC connection back to this docker-allocated IP
* ``-v /var/run/docker.sock:/var/run/docker.sock`` - makes the docker unix socket from the host available to the container. This is a common technique used to allow containers to launch other containers alongside itself.
* ``universe`` - use the imaged named 'universe' built above
* ``pytest`` - run 'pytest' in the container, which runs all the tests

At this point, you'll see a bunch of tests run and hopefully all pass.

To do some actual development work, you probably want to do another volume map from the universe repo on your host into the container, then shell in interactively:

```shell
docker run --privileged --rm -it -e DOCKER_NET_HOST=172.17.0.1 -v /var/run/docker.sock:/var/run/docker.sock -v (full path to cloned repo above):/usr/local/universe universe python
```

As you edit the files in your cloned git repo, they will be changed in your docker container and you'll be able to run them in python.

Note if you are using docker for Windows, you'll need to enable the relevant shared drive for this to work.

**Manage Docker as a non-root user:**

- Add the docker group if it doesn't already exist:

`sudo groupadd docker`

- Add the connected user "$USER" to the docker group. Change the user name to match your preferred user if you do not want to use your current user:

`sudo gpasswd -a $USER docker`

- Either do a `newgrp docker` or log out/in to activate the changes to groups.

- You can use

`docker run hello-world`

to check if you can run docker without sudo.


**Notes on installation**

* When installing ``universe``, you may see ``warning`` messages.  These lines occur when installing numpy and are normal.
* You'll need a ``go version`` of at least 1.5. Ubuntu 14.04 has an older Go, so you'll need to [upgrade](https://golang.org/doc/install) your Go installation.
* We run Python 3.5 internally, so the Python 3.5 variants will be much more thoroughly performance tested. Please let us know if you see any issues on 2.7.
* While we don't officially support Windows, we expect our code to be very close to working there. We'd be happy to take pull requests that take our Windows compatibility to 100%. In the meantime, the easiest way for Windows users to run universe is to use the alternate configuration described above.

#### Other Dependencies

* tensorflow
* cv2
* numpy
* gym

### Usage:

#### 1) Play with Coast_Racer trained by RL

- Run

```
docker run -p 5900:5900 -p 15900:15901 --cap-add SYS_ADMIN --ipc host --privileged quay.io/openai/universe.flashgames:0.20.7
```

- Run

```
python coast_racer_rl.py
```

#### 2) Play with DQN

**Training**

- Run

```
docker run -p 5900:5900 -p 15900:15901 --cap-add SYS_ADMIN --ipc host --privileged quay.io/openai/universe.flashgames:0.20.7
```

- Run

```
python coast_racer_dqn_train.py
```

**Test**

- Run

```
docker run -p 5900:5900 -p 15901:15901 --cap-add SYS_ADMIN --ipc host --privileged quay.io/openai/universe.flashgames:0.20.7
```

- Run

```
python coast_racer_dqn_test.py
```

Note that `coast_racer_dqn_train.py` doesn't render anything and as such, needs to be VNCd into to see whats happening.


### Video Demo

[Youtube Link](https://youtu.be/VdVA3od4tVs)

## Credits


