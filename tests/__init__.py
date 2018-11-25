# -*- coding: utf-8 -*-
# License: Apache License 2.0

import pkg_resources
from tempfile import NamedTemporaryFile

from wikipedia2vec.utils.wiki_dump_reader import WikiDumpReader
from wikipedia2vec.dump_db import DumpDB

dump_db = None
dump_db_file = None


def get_dump_db():
    return dump_db


def setUp():
    global dump_db, dump_db_file

    dump_file = pkg_resources.resource_filename(__name__,
                                                'test_data/enwiki-pages-articles-sample.xml.bz2')
    dump_reader = WikiDumpReader(dump_file)
    dump_db_file = NamedTemporaryFile()

    DumpDB.build(dump_reader, dump_db_file.name, 1, 1)
    dump_db = DumpDB(dump_db_file.name)


def tearDown():
    dump_db_file.close()
