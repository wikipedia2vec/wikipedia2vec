# -*- coding: utf-8 -*-

import click
import json

from wikipedia2vec import Wikipedia2Vec


@click.command()
@click.argument('model_file', type=click.Path(exists=True))
@click.option('--tensor-file', type=click.Path(), default='emb.tsv')
@click.option('--metadata-file', type=click.Path(), default='data.tsv')
@click.option('--config-file', type=click.Path(), default='config.json')
@click.option('--model-name', default='Wikipedia2Vec')
@click.option('--base-url', default='http://wikipedia2vec.github.io/projector_files/')
@click.option('--word-size', default=5000)
@click.option('--entity-size', default=5000)
def main(model_file, tensor_file, metadata_file, config_file, model_name, base_url, word_size,
         entity_size):
    model = Wikipedia2Vec.load(model_file)
    words = [w for w in sorted(model.dictionary.words(), key=lambda w: w.count,
                               reverse=True)[:word_size]]
    entities = [e for e in sorted(model.dictionary.entities(), key=lambda w: w.count,
                                  reverse=True)[:entity_size]]

    with open(tensor_file, mode='w', encoding='utf-8') as ten:
        with open(metadata_file, mode='w', encoding='utf-8') as meta:
            meta.write('item\tcount\n')
            for word in words:
                vector_str = '\t'.join(['%.2f' % v for v in model.get_vector(word)])
                ten.write(vector_str + '\n')
                meta.write('WORD/%s\t%d\n' % (word.text, word.count))

            for entity in entities:
                vector_str = '\t'.join(['%.2f' % v for v in model.get_vector(entity)])
                ten.write(vector_str + '\n')
                meta.write('ENTITY/%s\t%d\n' % (entity.title, entity.count))

    config_obj = {
        'embeddings': [
            {
                "tensorName": model_name,
                'tensorShape': [word_size + entity_size, model.syn0.shape[1]],
                "tensorPath": base_url + tensor_file,
                "metadataPath": base_url + metadata_file
            }
        ]
    }

    with open(config_file, mode='w', encoding='utf-8') as f:
        json.dump(config_obj, f, indent=2, sort_keys=True)


if __name__ == '__main__':
    main()
