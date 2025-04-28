#!/usr/bin/env python3

from syndicate import relay, turn, Symbol, Record, dataspace
from syndicate import patterns as P
from syndicate.during import During

@relay.service(name='coordinator')
@During().add_handler
def main(config):
    worker_ds = config['worker-dataspace'].embeddedValue
    turn.log.info('Got worker_ds %s', worker_ds)

    @dataspace.during(worker_ds, P.rec('worker-ready', P.CAPTURE))
    def worker_available(w):
        turn.log.info('Oh a worker is ready %s', w)
