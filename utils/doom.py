from vizdoom import *


def create_environment(config_file, wad_file):
    game = DoomGame()

    # Load the correct configuration
    game.load_config(config_file)

    # Load the correct scenario (in our case basic scenario)
    game.set_doom_scenario_path(wad_file)

    game.init()

    return game