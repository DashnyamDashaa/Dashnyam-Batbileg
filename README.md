hub is a command line tool that wraps `git` in order to extend it with extra
features and commands that make working with GitHub easier.

<!-- [contact Support](https://github.com/contact) -->

Usage
-----

``` sh
$ hub clone rtomayko/tilt
#=> git clone https://github.com/rtomayko/tilt.git

# or, if you prefer the SSH protocol:
$ git config --global hub.protocol ssh
$ hub clone rtomayko/tilt
#=> git clone git@github.com:rtomayko/tilt.git
```


Installation
------------

The `hub` executable has no dependencies, but since it was designed to wrap
`git`, it's recommended to have at least **git 1.7.3** or newer.

platform | manager | command to run
---------|---------|---------------
macOS, Linux | [Homebrew](https://docs.brew.sh/Installation) | `brew install hub`

