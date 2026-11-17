"""
Command-line interface for Curve2Flood.
"""

import argparse
import logging

from curve2flood import LOG, Curve2Flood_MainFunction

LOG_LEVELS = {
    'debug': logging.DEBUG,
    'info': logging.INFO,
    'warn': logging.WARNING,
    'error': logging.ERROR,
}

def set_log_level(log_level: str):
    """Set the level of Curve2Flood's logger: 'debug', 'info', 'warn' or 'error'."""
    level = LOG_LEVELS[log_level]
    LOG.setLevel(level)
    LOG.handlers[0].setLevel(level)
    LOG.info('Log Level set to ' + log_level)

def main():
    parser = argparse.ArgumentParser(description="Run Curve2Flood flood mapping tool.")
    parser.add_argument("input_file", help="Path to the input configuration file.")
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress all output except errors.")
    parser.add_argument('-l', '--log', type=str, help='Log Level',
                        default='warn', choices=list(LOG_LEVELS))
    args = parser.parse_args()

    set_log_level('error' if args.quiet else args.log)
    Curve2Flood_MainFunction(args.input_file, quiet=args.quiet)

if __name__ == "__main__":
    main()
