import sys, os
import pickle
from progress.bar import Bar
from midi_processor.processor import encode_midi

midi_in_path = 'midi_in'
encoded_in_path = 'encoded_in'

def find_files_by_extensions(root, exts=[]):

    def _has_ext(name):
        if not exts:
            return True
        name = name.lower()
        for ext in exts:
            if name.endswith(ext):
                return True
        return False

    for path, _, files in os.walk(root):
        for name in files:
            if _has_ext(name):
                yield os.path.join(path, name)

def preprocess_midi_folder(in_path = 'midi_in', out_path = 'encoded_in'):
    midi_paths = list(find_files_by_extensions(in_path, ['.mid', '.midi']))
    os.makedirs(out_path, exist_ok=True)

    out_fmt = '{}-{}.data'

    for path in Bar('Processing').iter(midi_paths):
        print(' ', end=f'[{path}]\n', flush=True)

        try:
            data = encode_midi(path)
        except KeyboardInterrupt:
            print(' Abort')
            return
        except EOFError:
            print('EOF Error')
            return

        with open('{}/{}.pickle'.format(out_path, path.split('\\')[-1]), 'wb') as f:
            pickle.dump(data, f)

if __name__ == '__main__':
    #in_path = midi_in_path if sys.argv[1]=='' else sys.argv[1]
    #out_path = encoded_in_path if sys.argv[2]=='' else sys.argv[2]

    preprocess_midi_folder()
