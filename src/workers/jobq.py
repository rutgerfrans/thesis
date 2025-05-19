from syndicate import relay, Record, patterns as P, dataspace, turn, Embedded
from syndicate.actor import ActiveFacet, Facet
from syndicate.during import During

Job = Record.makeConstructor('job', 'spec')
Worker = Record.makeConstructor('worker', 'entity')

InProgress = Record.makeConstructor('in-progress', 'spec entity handle')

def partition_list(xs, predicate):
    ys = []
    ns = []
    for x in xs:
        if predicate(x):
            ys.append(x)
        else:
            ns.append(x)
    return (ys, ns)

def truncated(s, length = 120):
    if len(s) >= length - 3:
        return s[:length - 3] + '...'
    else:
        return s

@relay.service(name='job-coordinator')
@During().add_handler
def main(data):
    coordination_ds  = data['dataspace'].embeddedValue

    free_workers = set() # Set of worker entity refs
    unclaimed_jobs = set() # Set of job specs
    in_progress = [] # List of (spec, entity ref, handle) pairings

    parent_facet = Facet.active
    turn.log.info('parent_facet %r', parent_facet)

    def make_pairing(spec, w):
        with ActiveFacet(parent_facet):
            handle = turn.publish(w, spec)
        turn.log.info('Pairing %s with %r at handle %r', truncated(repr(spec)), w, handle)
        in_progress.append(InProgress(spec, Embedded(w), handle))

    def retract_pairing(handle):
        turn.log.info('Retracting pairing handle %r', handle)
        with ActiveFacet(parent_facet):
            turn.retract(handle)

    def find_and_retract_pairings(predicate):
        nonlocal in_progress
        (matches, nonmatches) = partition_list(in_progress, predicate)
        for e in matches:
            retract_pairing(InProgress._handle(e))
        in_progress = nonmatches
        return matches

    def worker_available(entity):
        if unclaimed_jobs:
            make_pairing(unclaimed_jobs.pop(), entity)
        else:
            free_workers.add(entity)

    def job_available(spec):
        if free_workers:
            make_pairing(spec, free_workers.pop())
        else:
            unclaimed_jobs.add(spec)

    def summarize_state():
        turn.log.info('%d free workers, %d unclaimed jobs, %d jobs in progress',
                      len(free_workers),
                      len(unclaimed_jobs),
                      len(in_progress))

    @dataspace.during(coordination_ds, P.rec('job', P.CAPTURE))
    def new_job(spec):
        turn.log.info('New job %s', truncated(repr(spec)))
        job_available(spec)
        summarize_state()
        @turn.on_stop
        def job_retracted():
            turn.log.info('Job retracted %s', truncated(repr(spec)))
            unclaimed_jobs.discard(spec)
            for e in find_and_retract_pairings(lambda e: InProgress._spec(e) == spec):
                worker_available(InProgress._entity(e).embeddedValue)
            summarize_state()

    @dataspace.during(coordination_ds, P.rec('worker', P.CAPTURE))
    def new_worker(entity):
        if not isinstance(entity, Embedded): return
        entity = entity.embeddedValue
        turn.log.info('New worker %r', entity)
        worker_available(entity)
        summarize_state()
        @turn.on_stop
        def worker_retracted():
            turn.log.info('Worker retracted %r', entity)
            free_workers.discard(entity)
            for e in find_and_retract_pairings(lambda e: InProgress._entity(e).embeddedValue == entity):
                job_available(InProgress._spec(e))
            summarize_state()
