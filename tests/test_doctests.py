import doctest

import probkit.bloom
import probkit.hll


def load_tests(loader, tests, ignore):
    for mod in (probkit.bloom, probkit.hll):
        tests.addTests(doctest.DocTestSuite(mod))
    return tests
