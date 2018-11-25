# -*- coding: utf-8 -*-
# License: Apache License 2.0

from __future__ import unicode_literals
import numpy as np
import pickle
import six
import unittest
from tempfile import NamedTemporaryFile

from wikipedia2vec.dictionary import Dictionary, Entity
from wikipedia2vec.link_graph import LinkGraph
from wikipedia2vec.utils.tokenizer import get_default_tokenizer

from . import get_dump_db

from nose.tools import *


class TestLinkGraph(unittest.TestCase):
    def setUp(self):
        tokenizer = get_default_tokenizer('en')
        self.dictionary = Dictionary.build(get_dump_db(), tokenizer=tokenizer, lowercase=True,
                                           min_word_count=2, min_entity_count=1, pool_size=1,
                                           chunk_size=1, min_paragraph_len=5, category=True,
                                           disambi=False, progressbar=False)
        self.link_graph = LinkGraph.build(get_dump_db(), self.dictionary, pool_size=1, chunk_size=1,
                                          progressbar=False)

    def test_uuid_property(self):
        ok_(isinstance(self.link_graph.uuid, six.text_type))
        eq_(32, len(self.link_graph.uuid))

    def test_build_params_property(self):
        build_params = self.dictionary.build_params
        eq_(build_params['dump_db'], get_dump_db().uuid)
        ok_(build_params['dump_file'].endswith('enwiki-pages-articles-sample.xml.bz2'))
        ok_(isinstance(build_params['build_time'], float))
        ok_(build_params['build_time'] > 0)

    def test_neighbors(self):
        entity = self.dictionary.get_entity('Computer accessibility')
        neighbors = self.link_graph.neighbors(entity)
        eq_(115, len(neighbors))
        ok_(all(isinstance(entity, Entity) for entity in neighbors))

        titles = [e.title for e in neighbors]
        ok_('Accessibility' in titles)

    def test_save_load(self):
        def validate(obj):
            s1 = self.link_graph.serialize()
            s2 = obj.serialize()
            for key in s1.keys():
                if isinstance(s1[key], np.ndarray):
                    np.array_equal(s1[key], s2[key])
                else:
                    eq_(s1[key], s2[key])

        validate(LinkGraph.load(self.link_graph.serialize(), self.dictionary))

        with NamedTemporaryFile() as f:
            self.link_graph.save(f.name)
            validate(LinkGraph.load(f.name, self.dictionary))
