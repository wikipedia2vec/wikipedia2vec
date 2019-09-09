import logging
import re
from collections import defaultdict, Counter
from contextlib import closing
from multiprocessing.pool import Pool
import joblib
from marisa_trie import Trie, RecordTrie
from tqdm import tqdm

from data import normalize_text

logger = logging.getLogger(__name__)


class Mention(object):
    __slots__ = ('title', 'text', 'start', 'end', 'link_count', 'total_link_count', 'doc_count')

    def __init__(self, title, text, start, end, link_count, total_link_count, doc_count):
        self.title = title
        self.text = text
        self.start = start
        self.end = end
        self.link_count = link_count
        self.total_link_count = total_link_count
        self.doc_count = doc_count

    @property
    def span(self):
        return self.start, self.end

    @property
    def link_prob(self):
        if self.doc_count > 0:
            return min(1.0, self.total_link_count / self.doc_count)
        else:
            return 0.0

    @property
    def prior_prob(self):
        if self.total_link_count > 0:
            return min(1.0, self.link_count / self.total_link_count)
        else:
            return 0.0

    def __repr__(self):
        return f'<Mention {self.text} -> {self.title}>'


_dump_db = _tokenizer = _max_mention_length = _name_trie = None


class EntityLinker(object):
    def __init__(self, data_file, min_link_prob, min_prior_prob, min_link_count):
        self.min_link_prob = min_link_prob
        self.min_prior_prob = min_prior_prob
        self.min_link_count = min_link_count

        data = joblib.load(data_file)
        self.title_trie = data['title_trie']
        self.mention_trie = data['mention_trie']
        self.data_trie = data['data_trie']
        self.tokenizer = data['tokenizer']
        self.max_mention_length = data['max_mention_length']

    def detect_mentions(self, text):
        tokens = self.tokenizer.tokenize(text)
        end_offsets = frozenset(t.span[1] for t in tokens)

        ret = []
        cur = 0
        for token in tokens:
            start = token.span[0]
            if cur > start:
                continue

            for prefix in sorted(self.mention_trie.prefixes(text[start:start + self.max_mention_length]), key=len,
                                 reverse=True):
                end = start + len(prefix)
                if end in end_offsets:
                    matched = False

                    for title_id, link_count, total_link_count, doc_count in self.data_trie[prefix]:
                        if doc_count == 0 or total_link_count / doc_count < self.min_link_prob:
                            break
                        if link_count < self.min_link_count:
                            continue
                        if total_link_count == 0 or link_count / total_link_count < self.min_prior_prob:
                            continue

                        mention = Mention(self.title_trie.restore_key(title_id), prefix, start, end, link_count,
                                          total_link_count, doc_count)
                        ret.append(mention)
                        matched = True

                    if matched:
                        cur = end
                        break
        return ret

    @staticmethod
    def build(dump_db, tokenizer, out_file, min_link_prob, min_prior_prob, min_link_count, max_mention_length,
              pool_size, chunk_size):
        name_dict = defaultdict(Counter)

        logger.info('Iteration 1/2: Extracting all entity names...')
        with tqdm(total=dump_db.page_size(), mininterval=0.5) as pbar:
            initargs = (dump_db, tokenizer, max_mention_length)
            with closing(Pool(pool_size, initializer=EntityLinker._initialize_worker, initargs=initargs)) as pool:
                for ret in pool.imap_unordered(EntityLinker._extract_name_entity_pairs, dump_db.titles(),
                                               chunksize=chunk_size):
                    for text, title in ret:
                        name_dict[text][title] += 1
                    pbar.update()

        name_counter = Counter()

        disambi_matcher = re.compile(r'\s\(.*\)$')
        for title in dump_db.titles():
            text = normalize_text(disambi_matcher.sub('', title))
            name_dict[text][title] += 1
            name_counter[text] += 1

        for src, dest in dump_db.redirects():
            text = normalize_text(disambi_matcher.sub('', src))
            name_dict[text][dest] += 1
            name_counter[text] += 1

        logger.info('Iteration 2/2: Counting occurrences of entity names...')

        with tqdm(total=dump_db.page_size(), mininterval=0.5) as pbar:
            initargs = (dump_db, tokenizer, max_mention_length, Trie(name_dict.keys()))
            with closing(Pool(pool_size, initializer=EntityLinker._initialize_worker, initargs=initargs)) as pool:
                for names in pool.imap_unordered(EntityLinker._extract_name_occurrences, dump_db.titles(),
                                                 chunksize=chunk_size):
                    name_counter.update(names)
                    pbar.update()

        logger.info('Step 4/4: Building DB...')

        titles = frozenset([title for entity_counter in name_dict.values() for title in entity_counter.keys()])
        title_trie = Trie(titles)

        def item_generator():
            for name, entity_counter in name_dict.items():
                doc_count = name_counter[name]
                total_link_count = sum(entity_counter.values())

                if doc_count == 0:
                    continue

                link_prob = total_link_count / doc_count
                if link_prob < min_link_prob:
                    continue

                for title, link_count in entity_counter.items():
                    if link_count < min_link_count:
                        continue

                    prior_prob = link_count / total_link_count
                    if prior_prob < min_prior_prob:
                        continue

                    yield name, (title_trie[title], link_count, total_link_count, doc_count)

        data_trie = RecordTrie('<IIII', item_generator())
        mention_trie = Trie(data_trie.keys())

        joblib.dump(dict(title_trie=title_trie, mention_trie=mention_trie, data_trie=data_trie, tokenizer=tokenizer,
                         max_mention_length=max_mention_length), out_file)

    @staticmethod
    def _initialize_worker(dump_db, tokenizer, max_mention_length, name_trie=None):
        global _dump_db, _tokenizer, _max_mention_length, _name_trie
        _dump_db = dump_db
        _tokenizer = tokenizer
        _max_mention_length = max_mention_length
        _name_trie = name_trie

    @staticmethod
    def _extract_name_entity_pairs(title):
        ret = []
        for paragraph in _dump_db.get_paragraphs(title):
            for wiki_link in paragraph.wiki_links:
                if wiki_link.text and len(wiki_link.text) <= _max_mention_length:
                    ret.append((wiki_link.text, _dump_db.resolve_redirect(wiki_link.title)))
        return ret

    @staticmethod
    def _extract_name_occurrences(title):
        ret = []
        for paragraph in _dump_db.get_paragraphs(title):
            tokens = _tokenizer.tokenize(paragraph.text)
            end_offsets = frozenset(token.end for token in tokens)

            for token in tokens:
                start = token.start
                for prefix in _name_trie.prefixes(paragraph.text[start:start + _max_mention_length]):
                    if start + len(prefix) in end_offsets:
                        ret.append(prefix)
        return frozenset(ret)
