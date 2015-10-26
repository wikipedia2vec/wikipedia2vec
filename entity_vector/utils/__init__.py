# -*- coding: utf-8 -*-

import jnius_config
import pkg_resources

jnius_config.add_options('-Xrs', '-Xmx1024m')

jnius_config.set_classpath(
    pkg_resources.resource_filename(
        __name__, 'tokenizer/opennlp/opennlp-tools-1.5.3.jar'
    ),
    pkg_resources.resource_filename(
        __name__, 'ner/stanford_ner/stanford-ner-3.5.1.jar'
    ),
)
