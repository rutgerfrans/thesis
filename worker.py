#!/usr/bin/env python3

from syndicate import relay, turn, Symbol, Record
from syndicate.during import During

@relay.service(name='worker')
@During().add_handler
def main(config):
    turn.log.info('Got config %s', config)
