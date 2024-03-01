import math
import os
from collections import Counter
from hashlib import md5
import random
from typing import List, Tuple

import cv2

import numpy as np
import pandas as pd
from tensorflow.keras.utils import Sequence
from PIL import Image
from sklearn.preprocessing import LabelEncoder



Image.MAX_IMAGE_PIXELS = None
cv2.setNumThreads(0)


class PbNPatchesSequence(Sequence):

    def __init__(self, info, targets, batch_size, sample_size=None, patch_size=299, patches=1, positives=1, target_encoder=None, augment=False):
      self.info = info
      self.batch_size = batch_size
      self.sample_size = sample_size
      self.patch_size = patch_size
      self.patches = patches
      self.positives = positives
      self.augment = augment

      if not isinstance(targets, dict):
        self.multi_output = False
        targets = {"painter": targets}
        if target_encoder is not None and not isinstance(target_encoder, dict):
          target_encoder = {"painter": target_encoder}
      else:
        self.multi_output = True

      self.targets = targets
      self.target_encoder = target_encoder or {g: LabelEncoder().fit(y) for g, y in targets.items()}
      self.target_indices = {g: self.target_encoder[g].transform(y) for g, y in targets.items()}

    @property
    def painters(self):
      return self.target_encoder["painter"].classes_

    def __len__(self):
      return math.ceil(len(self.info) / self.batch_size)

    def __getitem__(self, idx):
      cv2.setNumThreads(0)

      low = idx * self.batch_size
      high = min(low + self.batch_size, len(self.info))

      batch_indices = list(range(low, high))
      batch_targets = self.target_indices["painter"][batch_indices]

      if self.positives > 1:
        pos_indices = []
        for target in batch_targets:
          indices_same_class, = np.where(target == self.target_indices["painter"])
          pos_indices += np.random.choice(indices_same_class, size=self.positives - 1).tolist()
        batch_indices += pos_indices

      batch_images = []
      for path in self.info.iloc[batch_indices].full_path:
        with Image.open(path) as img:
          img = img.convert("RGB")
          for _ in range(self.patches):
            batch_images.append(self.transform(img))

      batch_images = np.stack(batch_images, 0)
      batch_targets = {n: np.repeat(idxs[batch_indices].astype("int64"), self.patches)
                       for n, idxs in self.target_indices.items()}

      if not self.multi_output:
        batch_targets = batch_targets["painter"]

      return batch_images, batch_targets

    def transform(self, x):
      if self.sample_size:
        x = self._resize(x, self.sample_size, keep_aspect_ratio=True)
      x = self._random_crop(x, self.patch_size)
      x = self._resize(x, self.patch_size)
      if self.augment:
        x = self.augmentation(x)
      x = np.array(x, dtype="float32")
      x = self._normalize(x)

      return x

    def augmentation(self, x):
      x = self._random_horizontal_flip(x)
      return x

    def _random_crop(self, image, size):
      W, H = image.size

      left, top = np.random.rand(2)

      left = int(left * max(W-size, 0))
      top = int(top * max(H-size, 0))
      right = min(left + size, W)
      bottom = min(top + size, H)

      return image.crop((left, top, right, bottom))

    def _resize(self, image, size, keep_aspect_ratio=False):
      if keep_aspect_ratio:
        W, H = image.size
        ratio = H / W
        if W > H:
          sizes = int(size), int(size * ratio)
        else:
          sizes = int(size / ratio), int(size)
      else:
        sizes = (size, size)
      return image.resize(sizes, resample=Image.Resampling.BILINEAR)

    def _random_horizontal_flip(self, image: Image.Image):
      if np.random.randint(2) == 1:
        return image.transpose(Image.FLIP_LEFT_RIGHT)
      return image

    def _normalize(self, x):
      x /= 127.5
      x -= 1
      return x


def get_cleaned_dataset_path(data_dir):
  return os.path.join(data_dir, "all_data_info_cleaned.csv")


def get_dataset(args, override=False):
  info_path = os.path.join(args.data_dir, "all_data_info.csv")
  train_info_path = os.path.join(args.data_dir, "train_info.csv")
  info_cleaned_path = get_cleaned_dataset_path(args.data_dir)

  if not os.path.exists(info_cleaned_path) or override:
    info = load_info(info_path, train_info_path)
    info.set_index('filename', inplace=True)
    info = clean_pbn_dataset(info)
    info = info.reset_index()

    info["painter_index"] = LabelEncoder().fit_transform(info["painter"])
    info["style_index"] = LabelEncoder().fit_transform(info["style"])
    info["genre_index"] = LabelEncoder().fit_transform(info["genre"])
    info["full_path"] = info.apply(lambda x: os.path.join(args.data_dir, "train" if x.in_train else "test", x.filename), axis=1)

    info.to_csv(info_cleaned_path, index=False)

  info = pd.read_csv(info_cleaned_path)

  print(f"Dataset Load (artists={len(np.unique(info.painter))})")
  print("  Features = {", ", ".join(info.columns.values), "}")

  return info


def train_test_split(info, args):
  if args.data_split == "original":
    train, test = info[info.in_train], info[~info.in_train]

    train_painters = np.unique(train.painter)
    test_painters = np.unique(test.painter)

  elif args.data_split == "frequent":
    painters, n_paintings = np.unique(info.painter, return_counts=True)
    p = np.argsort(n_paintings)
    painters, n_paintings = painters[p], n_paintings[p]

    test_size = int(len(painters) * args.data_frequent_test_size)
    test_painters = painters[:test_size]
    train_painters = painters[test_size:]
    test_samples = info.painter.isin(test_painters)

    train, test = info[~test_samples], info[test_samples]
  else:
    raise ValueError(f"Unknown split {args.data_split}. Choices are 'original' and 'frequent'.")

  print(f"Dataset Split (split={args.data_split} train={len(train)} test={len(test)})")
  print(f"  Train (artists={len(train_painters)} avg-paintings={np.unique(train.painter, return_counts=True)[1].mean():.1f})")
  print(f"  Test  (artists={len(test_painters)} avg-paintings={np.unique(test.painter, return_counts=True)[1].mean():.1f})")

  return train, test


def load_info(all_info_path, train_info_path):
    info = pd.read_csv(all_info_path).rename(columns={'new_filename': 'filename', 'artist': 'artist_name'})
    train_info = (pd.read_csv(train_info_path)
                  .drop(columns='title style genre date'.split())
                  .rename(columns={'artist': 'artist_hash'})
                  .set_index('filename'))

    info = info.merge(train_info, on='filename', how='left')

    # Remove months, fic, circa and other textual
    # info from dates. Them cast them to float.
    info['date'] = (info.date
                    .str.lstrip(r'c\.|(January|February|March|April|May|September|October|November|December)\s')
                    .str.replace('fic', 'nan', regex=False)
                    .str.replace(r'^$', 'nan', regex=True)
                    .astype(float))

    # Clear artist name.
    # info['artist_name'] = info.artist_name.str.strip().astype('str') #mps

    # # Build artist-hash based on the artist name.
    # # Some artist hashes are missing. We will fill them by hashing the artist name.
    # codes = info[~info.artist_hash.isna()][['artist_name', 'artist_hash']].drop_duplicates()
    # codes = dict(zip(codes.artist_name, codes.artist_hash))

    # hashes = info.artist_hash.copy()
    # new_hashes = info.artist_name.map(lambda n: codes.get(n, md5(n.encode()).hexdigest()))
    # hashes[hashes.isna()] = new_hashes[hashes.isna()]
    # info['artist_hash'] = hashes #mps

    return info


def fix_names_with_digits(info):
  info.loc[(info['new_artist_name'] == '1') & (info['style'] == 'Minimalism'), 'new_artist_name'] = 'Yves Gaucher'
  info.loc[(info['new_artist_name'] == '1') & (info['style'] == 'Hard Edge Painting'), 'new_artist_name'] = 'Yves Gaucher'
  info.loc[(info['new_artist_name'] == '15') & (info['style'] == 'Color Field Painting'), 'new_artist_name'] = 'Yves Gaucher'
  info.loc[(info['new_artist_name'] == '2') & (info['style'] == 'Minimalism'), 'new_artist_name'] = 'Yves Gaucher'
  info.loc[(info['new_artist_name'] == '3') & (info['style'] == 'Minimalism'), 'new_artist_name'] = 'Yves Gaucher'
  info.loc[(info['new_artist_name'] == 'O/N/69') & (info['style'] == 'Minimalism'), 'new_artist_name'] = 'Yves Gaucher'
  #########
  info.loc[(info['new_artist_name'] == '1') & (info['style'] == 'Sōsaku hanga'), 'new_artist_name'] = 'Maki Haku'
  info.loc[(info['new_artist_name'] == '18') & (info['style'] == 'Sōsaku hanga'), 'new_artist_name'] = 'Maki Haku'
  info.loc[(info['new_artist_name'] == '27 (Stone)') & (info['style'] == 'Sōsaku hanga'), 'new_artist_name'] = 'Maki Haku'
  info.loc[(info['new_artist_name'] == '71') & (info['style'] == 'Sōsaku hanga'), 'new_artist_name'] = 'Maki Haku'
  #########
  info.loc[(info['new_artist_name'] == '1') & (info['style'] == 'Pop Art'), 'new_artist_name'] = 'Antonio Areal'
  #########
  info.loc[(info['new_artist_name'] == '1 (NOR)') & (info['style'] == 'Art Informel'), 'new_artist_name'] = 'Ernst Wilhelm Nay'
  info.loc[(info['new_artist_name'] == '2 (NOR)') & (info['style'] == 'Art Informel'), 'new_artist_name'] = 'Ernst Wilhelm Nay'
  info.loc[(info['new_artist_name'] == '21 (NOR)') & (info['style'] == 'Art Informel'), 'new_artist_name'] = 'Ernst Wilhelm Nay'
  #########
  info.loc[(info['new_artist_name'] == '10.3.07') & (info['style'] == 'Abstract Expressionism'), 'new_artist_name'] = 'John Hoyland'
  info.loc[(info['new_artist_name'] == '10.7.06') & (info['style'] == 'Abstract Expressionism'), 'new_artist_name'] = 'John Hoyland'
  info.loc[(info['new_artist_name'] == '11.7.07') & (info['style'] == 'Abstract Expressionism'), 'new_artist_name'] = 'John Hoyland'
  info.loc[(info['new_artist_name'] == '18.06.98') & (info['style'] == 'Abstract Expressionism'), 'new_artist_name'] = 'John Hoyland'
  info.loc[(info['new_artist_name'] == '20.3.08 (Clifford Dies)') & (info['style'] == 'Abstract Expressionism'), 'new_artist_name'] = 'John Hoyland'
  info.loc[(info['new_artist_name'] == '28.8.07') & (info['style'] == 'Abstract Expressionism'), 'new_artist_name'] = 'John Hoyland'
  info.loc[(info['new_artist_name'] == '5.11.07') & (info['style'] == 'Abstract Expressionism'), 'new_artist_name'] = 'John Hoyland'
  #########
  info.loc[(info['new_artist_name'] == '160 X 148') & (info['style'] == 'Minimalism'), 'new_artist_name'] = 'Martin Barre'
  #########
  info.loc[(info['new_artist_name'] == '1706), Lieutenant Admiral of Zeeland ') & (info['style'] == 'Baroque'), 'new_artist_name'] = 'Nicolaes Maes'
  #########
  info.loc[(info['new_artist_name'] == '2. Stanley Hawk') & (info['style'] == 'Naturalism'), 'new_artist_name'] = 'John James Audubon'
  #########
  info.loc[(info['new_artist_name'] == '23') & (info['style'] == 'Art Informel'), 'new_artist_name'] = 'Andre Lanskoy'
  info.loc[(info['new_artist_name'] == '36') & (info['style'] == 'Art Informel'), 'new_artist_name'] = 'Andre Lanskoy'
  info.loc[(info['new_artist_name'] == '39') & (info['style'] == 'Art Informel'), 'new_artist_name'] = 'Andre Lanskoy'
  info.loc[(info['new_artist_name'] == '75') & (info['style'] == 'Art Informel'), 'new_artist_name'] = 'Andre Lanskoy'
  info.loc[(info['new_artist_name'] == '76') & (info['style'] == 'Art Informel'), 'new_artist_name'] = 'Andre Lanskoy'
  info.loc[(info['new_artist_name'] == '77') & (info['style'] == 'Art Informel'), 'new_artist_name'] = 'Andre Lanskoy'
  info.loc[(info['new_artist_name'] == '7') & (info['style'] == 'Art Informel'), 'new_artist_name'] = 'Andre Lanskoy'
  #########
  info.loc[(info['new_artist_name'] == '4') & (info['style'] == 'Abstract Expressionism'), 'new_artist_name'] = 'Alvaro Lapa'
  #########
  info.loc[(info['new_artist_name'] == '500 Rakan temples') & (info['style'] == 'Ukiyo-e'), 'new_artist_name'] = 'Katsushika Hokusai'
  #########
  info.loc[(info['new_artist_name'] == '55 East Division Street') & (info['style'] == 'Magic Realism'), 'new_artist_name'] = 'Ivan Albright'
  #########
  info.loc[(info['new_artist_name'] == '69') & (info['style'] == 'Abstract Art'), 'new_artist_name'] = 'Fernando Lanhas'
  #########
  info.loc[(info['new_artist_name'] == '7th Avenue Style') & (info['style'] == 'Cubism'), 'new_artist_name'] = 'Stuart Davis'
  #########
  info.loc[(info['new_artist_name'] == '8 Janvier soir') & (info['style'] == 'Realism'), 'new_artist_name'] = 'Theophile Steinlen'
  info.loc[(info['new_artist_name'] == 'Fete du 8 Juin 1902  ') & (info['style'] == 'Art Nouveau (Modern)'), 'new_artist_name'] = 'Theophile Steinlen'
  #########
  info.loc[(info['new_artist_name'] == "Bakara Sûresi'nin ilk 4 âyeti") & (info['style'] == 'Ottoman Period'), 'new_artist_name'] = 'Sheikh Hamdullah'
  info.loc[(info['new_artist_name'] == "Fatiha Sûresi ve Bakara Sûresi'nin ilk 4 âyeti") & (info['style'] == 'Ottoman Period'), 'new_artist_name'] = 'Sheikh Hamdullah'
  #########
  info.loc[(info['new_artist_name'] == "Blackberry, pattern #388") & (info['style'] == 'Romanticism'), 'new_artist_name'] = 'William Morris'
  info.loc[(info['new_artist_name'] == "Hyacinth, pattern #480") & (info['style'] == 'Romanticism'), 'new_artist_name'] = 'William Morris'
  #########
  info.loc[(info['new_artist_name'] == "D4") & (info['style'] == 'Abstract Expressionism'), 'new_artist_name'] = 'Atsuko Tanaka'
  #########
  info.loc[(info['new_artist_name'] == "Getz/Gilberto #2") & (info['style'] == 'Lyrical Abstraction'), 'new_artist_name'] = 'Olga Albizu'
  #########
  info.loc[(info['new_artist_name'] == "Hanga Vol. 13") & (info['style'] == 'Sōsaku hanga'), 'new_artist_name'] = 'Koshiro Onchi'
  info.loc[(info['new_artist_name'] == "Hanga Vol.5") & (info['style'] == 'Sōsaku hanga'), 'new_artist_name'] = 'Koshiro Onchi'
  #########
  info.loc[(info['new_artist_name'] == "Hanga Vol.11") & (info['style'] == 'Sōsaku hanga'), 'new_artist_name'] = 'Unichi Hiratsuka'
  info.loc[(info['new_artist_name'] == "Yodo Vol.2") & (info['style'] == 'Sōsaku hanga'), 'new_artist_name'] = 'Unichi Hiratsuka'
  #########
  info.loc[(info['new_artist_name'] == "January 1958") & (info['style'] == 'Color Field Painting'), 'new_artist_name'] = 'Patrick Heron'
  info.loc[(info['new_artist_name'] == "July 1959") & (info['style'] == 'Post-Painterly Abstraction'), 'new_artist_name'] = 'Patrick Heron'
  #########
  info.loc[(info['new_artist_name'] == "les sorcières-1") & (info['style'] == 'Symbolism'), 'new_artist_name'] = 'Luc-Olivier Merson'
  #########
  info.loc[(info['new_artist_name'] == "no. 209") & (info['style'] == 'Abstract Art'), 'new_artist_name'] = 'Hans Richter'
  info.loc[(info['new_artist_name'] == "no. 301") & (info['style'] == 'Abstract Art'), 'new_artist_name'] = 'Hans Richter'
  info.loc[(info['new_artist_name'] == "no. 817B") & (info['style'] == 'Abstract Art'), 'new_artist_name'] = 'Hans Richter'
  info.loc[(info['new_artist_name'] == "no. 396") & (info['style'] == 'Dada'), 'new_artist_name'] = 'Hans Richter'
  info.loc[(info['new_artist_name'] == "no. 507") & (info['style'] == 'Dada'), 'new_artist_name'] = 'Hans Richter'
  #########
  info.loc[(info['new_artist_name'] == "Opus 2") & (info['style'] == 'Cubism'), 'new_artist_name'] = 'Victor Servranckx'
  #########
  info.loc[(info['new_artist_name'] == "Re Rom 337") & (info['style'] == 'Concretism'), 'new_artist_name'] = 'Aurelie Nemours'
  #########
  info.loc[(info['new_artist_name'] == "Reunion 2") & (info['style'] == 'Neo-Romanticism'), 'new_artist_name'] = 'Stanley Spencer'
  info.loc[(info['new_artist_name'] == "Waking Up 2") & (info['style'] == 'Neo-Romanticism'), 'new_artist_name'] = 'Stanley Spencer'
  #########
  info.loc[(info['new_artist_name'] == 'Revista Portugueza", nº 162, Ano IV de 23 de Agosto') & (info['style'] == 'Art Nouveau (Modern)'), 'new_artist_name'] = 'Emmerico Nunes'
  #########
  info.loc[(info['new_artist_name'] == "S. Sebastien 3") & (info['style'] == 'Naïve Art (Primitivism)'), 'new_artist_name'] = 'Joaquim Rodrigo'
  #########
  info.loc[(info['new_artist_name'] == "SF78-235") & (info['style'] == 'Color Field Painting'), 'new_artist_name'] = 'Sam Francis'
  #########
  info.loc[(info['new_artist_name'] == "Total Eclipse / Rotorelief n°12") & (info['style'] == 'Op Art'), 'new_artist_name'] = 'Marcel Duchamp'
  #########
  info.loc[(info['new_artist_name'] == "Two Nazi Soldiers Mistreating Jews 1942") & (info['style'] == 'Expressionism'), 'new_artist_name'] = 'Marcel Janco'
  #########
  info.loc[(info['new_artist_name'] == "Urban Landscape, No. 27") & (info['style'] == 'Abstract Expressionism'), 'new_artist_name'] = 'John Cage'
  #########
  info.loc[(info['new_artist_name'] == "x2 + bx + c rouge-vert") & (info['style'] == 'Neoplasticism'), 'new_artist_name'] = 'Georges Vantongerloo'
  #########
  info.loc[(info['new_artist_name'] == "∞ / détail 4875812") & (info['style'] == 'Conceptual Art'), 'new_artist_name'] = 'Roman Opalka'
  info.loc[(info['new_artist_name'] == "∞ / détail 4894231") & (info['style'] == 'Conceptual Art'), 'new_artist_name'] = 'Roman Opalka'
  info.loc[(info['new_artist_name'] == "∞ / détail 4914800") & (info['style'] == 'Conceptual Art'), 'new_artist_name'] = 'Roman Opalka'
  info.loc[(info['new_artist_name'] == "∞ / détail 511130-512739") & (info['style'] == 'Conceptual Art'), 'new_artist_name'] = 'Roman Opalka'
  info.loc[(info['new_artist_name'] == "∞ /carte de voyage détail 2333587") & (info['style'] == 'Conceptual Art'), 'new_artist_name'] = 'Roman Opalka'
  info.loc[(info['new_artist_name'] == "∞ /detail 868149-893746") & (info['style'] == 'Conceptual Art'), 'new_artist_name'] = 'Roman Opalka'

  return info


def get_artist_per_hash(mapping, counter):
  map_corrected = {}

  for hash in mapping:
    v_count = 0
    for name in mapping[hash]:
      count = counter[name]
      if count > v_count:
        v_count = count
        v_name = name

    if hash not in map_corrected:
      map_corrected[hash] = v_name
    else:
      print('An inconsistency has been found!')

  return map_corrected


## Creates a map, associating a unique hash with all unique artist names that belong to it
def create_name_hash_map(info):
  name_hash_map = {}
  for i, row in info.iterrows():
    hash = row['artist_hash']
    artist = row['new_artist_name']

    if hash is np.nan:
      hash = 'NAN'
    if hash not in name_hash_map:
      name_hash_map[hash] = []

    if artist not in name_hash_map[hash]:
      name_hash_map[hash].append(row['new_artist_name'])

  ## Separates the map according to the amount of unique artist names per hash
  hash_nan = name_hash_map['NAN']
  hash_count_one = {h:v for h,v in name_hash_map.items() if len(v) <= 1}
  hash_count_more = {h:v for h,v in name_hash_map.items() if len(v) > 1 and h != 'NAN'}

  return name_hash_map, hash_nan, hash_count_one, hash_count_more


def fix_hashes_with_several_names(info, hash_count_more):
  ## Counts the frequency of each artist in the dataset
  all_artist_names = info['new_artist_name'].values
  counts_artists = Counter(all_artist_names)

  ## Gets the artist name per hash, for those hashes that have more than one name
  hash_more_correction = get_artist_per_hash(hash_count_more, counts_artists)
  ## Corrects names
  for hash in hash_more_correction:
    info.loc[(info['artist_hash'] == hash), 'new_artist_name'] = hash_more_correction[hash]

  return info


def fix_artist_without_hash(info):
  info.loc[(info['new_artist_name'] == 'At Les Ambassadeurs'), 'new_artist_name'] = 'Edgar Degas'
  ###
  info.loc[(info['new_artist_name'] == 'Vase with Twelve Sunflowers '), 'new_artist_name'] = 'Vincent van Gogh'
  ###
  info.loc[(info['new_artist_name'] == 'Entree Dans Les Geoles Allemandes '), 'new_artist_name'] = 'Theophile Steinlen'
  info.loc[(info['new_artist_name'] == 'en longeur '), 'new_artist_name'] = 'Theophile Steinlen'
  ###
  info.loc[(info['new_artist_name'] == 'St. James the Great and St. Clare'), 'new_artist_name'] = 'Robert Campin'
  info.loc[(info['new_artist_name'] == 'Joseph as a medieval carpenter'), 'new_artist_name'] = 'Robert Campin'
  info.loc[(info['new_artist_name'] == 'The Cope of Saint John'), 'new_artist_name'] = 'Robert Campin'
  info.loc[(info['new_artist_name'] == 'The Cope of the Virgin Mary'), 'new_artist_name'] = 'Robert Campin'
  info.loc[(info['new_artist_name'] == 'The Donors'), 'new_artist_name'] = 'Robert Campin'
  info.loc[(info['new_artist_name'] == 'St. John the Baptist and the Donor, Heinrich Von Wer'), 'new_artist_name'] = 'Robert Campin'
  ###
  info.loc[(info['new_artist_name'] == 'the Death of Raymond Lulle'), 'new_artist_name'] = 'Salvador Dali'
  info.loc[(info['new_artist_name'] == 'the Central Element)'), 'new_artist_name'] = 'Salvador Dali'
  info.loc[(info['new_artist_name'] == 'Continuum of the Four Buttocks or Five Rhinoceros Horns Making a Virgin or Birth of a Deity'), 'new_artist_name'] = 'Salvador Dali'
  info.loc[(info['new_artist_name'] == 'Saint Sebastian'), 'new_artist_name'] = 'Salvador Dali'
  info.loc[(info['new_artist_name'] == 'Sir James Dunn Seated'), 'new_artist_name'] = 'Salvador Dali'
  info.loc[(info['new_artist_name'] == "Illustration for 'Memories of Surrealism'"), 'new_artist_name'] = 'Salvador Dali'
  ###
  info.loc[(info['new_artist_name'] == 'Bow'), 'new_artist_name'] = 'Gino Severini'
  ###
  info.loc[(info['new_artist_name'] == 'Colombina'), 'new_artist_name'] = 'Serge Sudeikin'
  info.loc[(info['new_artist_name'] == 'Harlequin'), 'new_artist_name'] = 'Serge Sudeikin'
  ###
  info.loc[(info['new_artist_name'] == 'Night'), 'new_artist_name'] = 'Koloman Moser'
  info.loc[(info['new_artist_name'] == 'Morning'), 'new_artist_name'] = 'Koloman Moser'
  ###
  info.loc[(info['new_artist_name'] == 'The Adoration of the Magi'), 'new_artist_name'] = 'Hans Holbein the Younger'
  ###
  info.loc[(info['new_artist_name'] == 'set design '), 'new_artist_name'] = 'Leon Bakst'
  info.loc[(info['new_artist_name'] == 'costume for Ida Rubinstein as Helene'), 'new_artist_name'] = 'Leon Bakst'
  ###
  info.loc[(info['new_artist_name'] == 'The Appearance of a concubine of the Bunka Era'), 'new_artist_name'] = 'Tsukioka Yoshitoshi'
  info.loc[(info['new_artist_name'] == 'The appearance of a-Fukagawa-Nakamichi-Geisha-of-the-Tempo-era'), 'new_artist_name'] = 'Tsukioka Yoshitoshi'
  info.loc[(info['new_artist_name'] == 'The appearance of a ‘castle-toppler’ of the Tempo era'), 'new_artist_name'] = 'Tsukioka Yoshitoshi'
  info.loc[(info['new_artist_name'] == 'The appearance of a courtesan during the Kaei era'), 'new_artist_name'] = 'Tsukioka Yoshitoshi'
  info.loc[(info['new_artist_name'] == 'The appearance of a Kyoto geisha of the Kansei era'), 'new_artist_name'] = 'Tsukioka Yoshitoshi'
  info.loc[(info['new_artist_name'] == 'The appearance of a courtesan of the Meiji era'), 'new_artist_name'] = 'Tsukioka Yoshitoshi'
  info.loc[(info['new_artist_name'] == 'a court lady of the Kyowa era'), 'new_artist_name'] = 'Tsukioka Yoshitoshi'
  info.loc[(info['new_artist_name'] == 'The appearance of a maiden of the Koka era'), 'new_artist_name'] = 'Tsukioka Yoshitoshi'
  info.loc[(info['new_artist_name'] == 'The Appearance of a Young Lady from Nagoya During the Ansei era'), 'new_artist_name'] = 'Tsukioka Yoshitoshi'
  ###
  info.loc[(info['new_artist_name'] == '''happiness can not see".In a rich peasant's hut'''), 'new_artist_name'] = 'Konstantin Yuon'
  ###
  info.loc[(info['new_artist_name'] == 'Boats on the Dogana'), 'new_artist_name'] = 'Oskar Kokoschka'
  ###
  info.loc[(info['new_artist_name'] == 'friend of travelers'), 'new_artist_name'] = 'Nicholas Roerich'
  ###
  info.loc[(info['new_artist_name'] == '- Franz Kline'), 'new_artist_name'] = 'Franz Kline'
  ###
  info.loc[(info['new_artist_name'] == 'Three Men'), 'new_artist_name'] = 'William H. Johnson'
  ###
  info.loc[(info['new_artist_name'] == 'the oldest child takes care of brothers and sisters in the absence of parents '), 'new_artist_name'] = 'Ferdinand Georg Waldmüller'
  ###
  info.loc[(info['new_artist_name'] == 'Dorfstrasse'), 'new_artist_name'] = 'Alexej von Jawlensky'
  info.loc[(info['new_artist_name'] == 'Christ'), 'new_artist_name'] = 'Alexej von Jawlensky'
  ###
  info.loc[(info['new_artist_name'] == 'Effect of Snow'), 'new_artist_name'] = 'Camille Pissarro'
  ###
  info.loc[(info['new_artist_name'] == 'No. I'), 'new_artist_name'] = "Georgia O'Keeffe"
  info.loc[(info['new_artist_name'] == 'Red with Yellow'), 'new_artist_name'] = "Georgia O'Keeffe"
  info.loc[(info['new_artist_name'] == 'Chestnut Tree'), 'new_artist_name'] = "Georgia O'Keeffe"
  info.loc[(info['new_artist_name'] == 'Hills to the left'), 'new_artist_name'] = "Georgia O'Keeffe"
  ###
  info.loc[(info['new_artist_name'] == 'Appledore, Isles of Shoals'), 'new_artist_name'] = 'Childe Hassam'
  info.loc[(info['new_artist_name'] == 'Villiers-le-Bel'), 'new_artist_name'] = 'Childe Hassam'
  ###
  info.loc[(info['new_artist_name'] == 'Dua'), 'new_artist_name'] = 'Mustafa Rakim'
  info.loc[(info['new_artist_name'] == 'Muhammed (A.S.)'), 'new_artist_name'] = 'Mustafa Rakim'
  info.loc[(info['new_artist_name'] == 'Ayet-i Kerîme'), 'new_artist_name'] = 'Mustafa Rakim'
  info.loc[((info['new_artist_name'] == 'Hadis-i Şerîf') & (info['artist_group'] == 'train_and_test')), 'new_artist_name'] = 'Mustafa Rakim'
  ###
  info.loc[(info['new_artist_name'] == 'Fábula'), 'new_artist_name'] = 'El Greco'
  ###
  info.loc[(info['new_artist_name'] == 'Time Bomb'), 'new_artist_name'] = 'Martin Disler'
  ###
  info.loc[(info['new_artist_name'] == 'The Cigarette '), 'new_artist_name'] = 'Robert Henri'
  info.loc[(info['new_artist_name'] == 'Concarneau'), 'new_artist_name'] = 'Robert Henri'
  info.loc[(info['new_artist_name'] == 'Seviliana (also known as Dancer with Castanet) '), 'new_artist_name'] = 'Robert Henri'
  info.loc[(info['new_artist_name'] == 'Monhegan'), 'new_artist_name'] = 'Robert Henri'
  ###
  info.loc[(info['new_artist_name'] == 'The Yellow River Breaches its Course'), 'new_artist_name'] = 'Ma Yuan'
  info.loc[(info['new_artist_name'] == 'The Waving Surface of the Autumn Flood'), 'new_artist_name'] = 'Ma Yuan'
  ###
  info.loc[(info['new_artist_name'] == 'An Exterior'), 'new_artist_name'] = 'James McNeill Whistler'
  info.loc[(info['new_artist_name'] == 'Canal'), 'new_artist_name'] = 'James McNeill Whistler'
  info.loc[(info['new_artist_name'] == 'Builder of Temples'), 'new_artist_name'] = 'James McNeill Whistler'
  info.loc[(info['new_artist_name'] == 'Mouth of the Thames'), 'new_artist_name'] = 'James McNeill Whistler'
  info.loc[(info['new_artist_name'] == 'The Golden Bay '), 'new_artist_name'] = 'James McNeill Whistler'
  info.loc[(info['new_artist_name'] == 'Bognor'), 'new_artist_name'] = 'James McNeill Whistler'
  ###
  info.loc[(info['new_artist_name'] == 'Time Flies'), 'new_artist_name'] = 'Frida Kahlo'
  ###
  info.loc[(info['new_artist_name'] == 'Sts Eligius and Catherine'), 'new_artist_name'] = 'Peter Paul Rubens'
  ###
  info.loc[(info['new_artist_name'] == 'right wing of The Bourbon Altarpiece'), 'new_artist_name'] = 'Jean Hey'
  ###
  info.loc[(info['new_artist_name'] == 'draft'), 'new_artist_name'] = 'Eugene Delacroix'
  ###
  info.loc[(info['new_artist_name'] == 'A Thousand Roots Did Die With Thee'), 'new_artist_name'] = 'Hans Hofmann'
  ###
  info.loc[(info['new_artist_name'] == 'Hollyhocks'), 'new_artist_name'] = 'Eugene Boudin'
  info.loc[(info['new_artist_name'] == 'Finistere'), 'new_artist_name'] = 'Eugene Boudin'
  ###
  info.loc[(info['new_artist_name'] == 'The Singer Fate Fashioned to Her Liking'), 'new_artist_name'] = 'Howard Pyle'
  ###
  info.loc[(info['new_artist_name'] == 'Confirmation'), 'new_artist_name'] = 'Giuseppe Maria Crespi'
  ###
  info.loc[(info['new_artist_name'] == 'Chepstow Castle'), 'new_artist_name'] = 'John Martin'
  ###
  info.loc[(info['new_artist_name'] == 'Church of the Covenant (Boston) '), 'new_artist_name'] = 'Louis Comfort Tiffany'
  ###
  info.loc[(info['new_artist_name'] == 'seated person'), 'new_artist_name'] = 'Umberto Boccioni'
  ###
  info.loc[(info['new_artist_name'] == 'central panel'), 'new_artist_name'] = 'Maerten van Heemskerck'
  info.loc[(info['new_artist_name'] == 'right panel'), 'new_artist_name'] = 'Maerten van Heemskerck'
  ###
  info.loc[(info['new_artist_name'] == 'The Pont Molle'), 'new_artist_name'] = 'Claude Lorrain'
  ###
  info.loc[(info['new_artist_name'] == 'Seraphs'), 'new_artist_name'] = 'Wilhelm Kotarbinski'
  ###
  info.loc[(info['new_artist_name'] == 'The Damned Consigned to Hell)'), 'new_artist_name'] = 'Luca Signorelli'
  ###
  info.loc[(info['new_artist_name'] == 'Piute Indian at Bridal Veil Falls, Yosemite'), 'new_artist_name'] = 'Thomas Hill'
  ###
  info.loc[(info['new_artist_name'] == 'Capri'), 'new_artist_name'] = 'Henrique Pousao'
  ###
  info.loc[(info['new_artist_name'] == 'detail'), 'new_artist_name'] = 'Nakahara Nantenbo'
  ###
  info.loc[(info['new_artist_name'] == 'The scorpion'), 'new_artist_name'] = 'Stanley Spencer'
  info.loc[(info['new_artist_name'] == 'Awaking'), 'new_artist_name'] = 'Stanley Spencer'
  info.loc[(info['new_artist_name'] == 'The foxes have holes'), 'new_artist_name'] = 'Stanley Spencer'
  info.loc[(info['new_artist_name'] == 'Driven by the spirit'), 'new_artist_name'] = 'Stanley Spencer'
  info.loc[(info['new_artist_name'] == 'Waking up'), 'new_artist_name'] = 'Stanley Spencer'
  ###
  info.loc[(info['new_artist_name'] == 'Hadis-i Şerîfler'), 'new_artist_name'] = 'Sheikh Hamdullah'
  info.loc[((info['new_artist_name'] == 'Hadis-i Şerîf') & (info['artist_group'] == 'test_only')), 'new_artist_name'] = 'Sheikh Hamdullah'
  ###
  info.loc[(info['new_artist_name'] == 'A Corner in the Alhambra '), 'new_artist_name'] = 'Tom Roberts'
  ###
  info.loc[(info['new_artist_name'] == 'from Beauty and the Beast'), 'new_artist_name'] = 'Edmund Dulac'
  info.loc[(info['new_artist_name'] == 'The Nightingale'), 'new_artist_name'] = 'Edmund Dulac'
  ###
  info.loc[(info['new_artist_name'] == 'Bristol Quay'), 'new_artist_name'] = 'Eric Ravilious'
  ###
  info.loc[(info['new_artist_name'] == 'Africa'), 'new_artist_name'] = 'Tihamer Gyarmathy'
  ###
  info.loc[(info['new_artist_name'] == 'Prospero, Miranda and Caliban Spy '), 'new_artist_name'] = 'Thomas Jones'
  ###
  info.loc[(info['new_artist_name'] == 'les trois sorcières'), 'new_artist_name'] = 'Luc-Olivier Merson'
  info.loc[(info['new_artist_name'] == 'Lady Macbeth somnambule'), 'new_artist_name'] = 'Luc-Olivier Merson'
  info.loc[(info['new_artist_name'] == 'Hécate et les trois sorcières'), 'new_artist_name'] = 'Luc-Olivier Merson'
  info.loc[(info['new_artist_name'] == 'assassinat de Banquo'), 'new_artist_name'] = 'Luc-Olivier Merson'
  info.loc[(info['new_artist_name'] == 'les trois sorcières agenouillées'), 'new_artist_name'] = 'Luc-Olivier Merson'
  info.loc[(info['new_artist_name'] == 'entrée des sorcières'), 'new_artist_name'] = 'Luc-Olivier Merson'
  ###
  info.loc[(info['new_artist_name'] == 'Baby in Blue'), 'new_artist_name'] = 'Twins Seven Seven'
  ###
  info.loc[(info['new_artist_name'] == 'The holiday of harvest '), 'new_artist_name'] = 'Mykhailo Boychuk'
  ###
  info.loc[(info['new_artist_name'] == 'Mirror of the Ages'), 'new_artist_name'] = 'Toyohara Chikanobu'
  ###
  info.loc[(info['new_artist_name'] == 'The Epic of American Civilization'), 'new_artist_name'] = 'Jose Clemente Orozco'
  ###
  info.loc[(info['new_artist_name'] == 'Summer'), 'new_artist_name'] = 'Toshi Yoshida'
  ###
  info.loc[(info['new_artist_name'] == 'Violet'), 'new_artist_name'] = 'Paul Jacoulet'
  info.loc[(info['new_artist_name'] == 'Blue'), 'new_artist_name'] = 'Paul Jacoulet'
  info.loc[(info['new_artist_name'] == 'Orange'), 'new_artist_name'] = 'Paul Jacoulet'
  info.loc[(info['new_artist_name'] == 'Red'), 'new_artist_name'] = 'Paul Jacoulet'
  ###
  info.loc[(info['new_artist_name'] == 'Aberdeen '), 'new_artist_name'] = 'Tom Scott'
  ###
  info.loc[(info['new_artist_name'] == 'leuchtrot warm'), 'new_artist_name'] = 'Rupprecht Geiger'
  ###
  info.loc[(info['new_artist_name'] == 'Early Morning '), 'new_artist_name'] = 'Samuel Palmer'
  ###
  info.loc[(info['new_artist_name'] == 'Modality Series'), 'new_artist_name'] = 'Jock Macdonald'
  ###
  info.loc[(info['new_artist_name'] == 'right'), 'new_artist_name'] = 'Charles Rennie Mackintosh'
  ###
  info.loc[(info['new_artist_name'] == ' François Leclerc du Tremblay (detail)'), 'new_artist_name'] = 'Jean-Leon Gerome'
  ###
  info.loc[(info['new_artist_name'] == 'Country Road  '), 'new_artist_name'] = 'Clarence Holbrook Carter'
  ###
  info.loc[(info['new_artist_name'] == 'vent lumiere'), 'new_artist_name'] = 'Gustave Singier'
  info.loc[(info['new_artist_name'] == 'Port, Mistral, Nuit '), 'new_artist_name'] = 'Gustave Singier'
  info.loc[(info['new_artist_name'] == 'nuit'), 'new_artist_name'] = 'Gustave Singier'
  ###
  info.loc[(info['new_artist_name'] == 'In Paris'), 'new_artist_name'] = 'Carlos Botelho'
  ###
  info.loc[(info['new_artist_name'] == 'Nice'), 'new_artist_name'] = 'Joaquim Rodrigo'
  ###
  info.loc[(info['new_artist_name'] == 'La Cruche saigne'), 'new_artist_name'] = 'Louis Soutter'
  ###
  info.loc[(info['new_artist_name'] == 'GRB I'), 'new_artist_name'] = 'Yves Gaucher'
  ###
  info.loc[(info['new_artist_name'] == 'Lake Superior Sketch XXXIV'), 'new_artist_name'] = 'Lawren Harris'
  ###
  info.loc[(info['new_artist_name'] == 'Lady in the Hat VI'), 'new_artist_name'] = 'Albin Brunovsky'
  ###
  info.loc[(info['new_artist_name'] == 'Untitled XIX'), 'new_artist_name'] = 'Piero Dorazio'
  ###
  info.loc[(info['new_artist_name'] == 'Grey'), 'new_artist_name'] = 'Perle Fine'
  ###
  info.loc[(info['new_artist_name'] == 'Man of Peace'), 'new_artist_name'] = 'M.F. Husain'
  ###
  info.loc[(info['new_artist_name'] == 'Gold and Brown'), 'new_artist_name'] = 'Lygia Pape'
  info.loc[(info['new_artist_name'] == 'Gold and Blue (Sertão Carioca)'), 'new_artist_name'] = 'Lygia Pape'
  ###
  info.loc[(info['new_artist_name'] == 'Ou Kaapstad'), 'new_artist_name'] = 'Gregoire Boonzaier'
  ###
  info.loc[(info['new_artist_name'] == 'Wintry Sky'), 'new_artist_name'] = 'Yamamura Toyonari'
  ###
  info.loc[(info['new_artist_name'] == '(Peaches and Grapes) '), 'new_artist_name'] = 'Yasuo Kuniyoshi'
  info.loc[(info['new_artist_name'] == '(On the Wire) '), 'new_artist_name'] = 'Yasuo Kuniyoshi'
  ###
  info.loc[(info['new_artist_name'] == 'Self Portrait'), 'new_artist_name'] = 'Abidin Dino'
  ###
  info.loc[(info['new_artist_name'] == 'Le Torrent vert'), 'new_artist_name'] = 'Alfred Manessier'
  ###
  info.loc[(info['new_artist_name'] == 'Tierra'), 'new_artist_name'] = 'Albert Rafols-Casamada'

  return info


def fix_hashes(info):
  ## Gets the cases where artist hash is not NAN, removing duplicate entries
  codes = info[~info.artist_hash.isna()][['new_artist_name', 'artist_hash']].drop_duplicates()
  ## Creates a dict in which the key is the artist name and the value is tha hash
  codes = dict(zip(codes.new_artist_name, codes.artist_hash))
  ## Copies the original hashes
  hashes = info.artist_hash.copy()
  ## Gets the hashes for those artists that already have it or create a new one for those artists that don't have it
  new_hashes = info.new_artist_name.map(lambda n: codes.get(n, md5(n.encode()).hexdigest()))
  ## Replaces the NAN with the created hashes
  hashes[hashes.isna()] = new_hashes[hashes.isna()]
  ## Updates the artist hash column with all the hashes
  info['artist_hash'] = hashes

  return info


def fix_remaining_erros(info):
  info.loc[(info['new_artist_name'] == 'Setsu Getsu Ka') & (info['style'] == 'Ukiyo-e'), 'new_artist_name'] = 'Toyohara Chikanobu'
  ###
  info.loc[(info['new_artist_name'] == 'Stare gniazdo') & (info['style'] == 'Impressionism'), 'new_artist_name'] = 'Ferdynand Ruszczyc'
  ###
  info.loc[(info['new_artist_name'] == 'Lake George, N.Y.') & (info['style'] == 'Precisionism'), 'new_artist_name'] = "Georgia O'Keeffe"
  ###
  info.loc[(info['new_artist_name'] == 'Rote Dächer') & (info['style'] == 'Expressionism'), 'new_artist_name'] = "Alexej von Jawlensky"
  ###
  info.loc[(info['new_artist_name'] == 'Storm Sea ') & (info['style'] == 'American Realism'), 'new_artist_name'] = "Robert Henri"
  info.loc[(info['new_artist_name'] == 'Anthony Lavelle') & (info['style'] == 'American Realism'), 'new_artist_name'] = "Robert Henri"
  # ###
  info.loc[(info['new_artist_name'] == 'Fuzoku Sanjuniso') & (info['style'] == 'Ukiyo-e'), 'new_artist_name'] = "Tsukioka Yoshitoshi"
  # ###
  info.loc[(info['new_artist_name'] == 'Ordination') & (info['style'] == 'Baroque'), 'new_artist_name'] = "Giuseppe Maria Crespi"
  # ###
  info.loc[(info['new_artist_name'] == 'Hana Kurabe') & (info['style'] == 'Ukiyo-e'), 'new_artist_name'] = "Shibata Zeshin"
  # ###
  info.loc[(info['new_artist_name'] == 'Petits Fieux ') & (info['style'] == 'Art Nouveau (Modern)'), 'new_artist_name'] = "Theophile Steinlen"
  # ###
  info.loc[(info['new_artist_name'] == 'Weiss auf Schwarz') & (info['style'] == 'Op Art'), 'new_artist_name'] = "Walter Leblanc"
  # ###
  info.loc[(info['new_artist_name'] == 'Plakat') & (info['style'] == 'Surrealism'), 'new_artist_name'] = "Paul Wunderlich"

  return info


def clean_pbn_dataset(info):
  info['new_artist_name'] = info['artist_name']
  info = fix_names_with_digits(info)
  name_hash_map, hash_nan, hash_count_one, hash_count_more = create_name_hash_map(info)
  info = fix_hashes_with_several_names(info, hash_count_more)
  info = fix_artist_without_hash(info)
  info = fix_remaining_erros(info)
  info = fix_hashes(info)

  return info.rename(columns={"new_artist_name": "painter"})


def load_features_file(features_dir: str, extractor_name: str, patches: int = None, parts: int = 4) -> Tuple[List[str], np.ndarray]:
  print(f"Attempting to load features from {features_dir}{extractor_name}...", flush=True)

  for part in range(parts):
    features_path = os.path.join(features_dir, extractor_name + f".part-{part}.npz")

    fnames = []
    features = []

    if not os.path.exists(features_path):
      print(f"  part {part} of {extractor_name} ({features_path}) missing.")
      continue

    # print(f"Loading features in {features_path}", flush=True)
    blob = np.load(features_path, allow_pickle=True)
    b_fnames = blob["filenames"]
    b_feats = blob["features"]

    if patches:
      b_feats = b_feats[:, :patches]

    print(f"  part {part} loaded: names=({len(b_fnames)}) "
          f"features=(shape={b_feats.shape}, dtype={b_feats.dtype})",
          flush=True)

    fnames.append(b_fnames)
    features.append(b_feats)

    unames, ucounts = np.unique(b_fnames, return_counts=True)
    print(f"fnames={len(fnames)} unames={len(unames)} "
          f"ucounts-min={ucounts.min()} ucounts-max={ucounts.max()}", flush=True)

  fnames, features = (np.concatenate(e, axis=0) for e in (fnames, features))

  return fnames, features
