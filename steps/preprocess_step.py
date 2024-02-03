import os
import concurrent

from PIL import Image

from datasets.pbn import get_cleaned_dataset_path


def reduce_massive_images(dataset: "pd.DataFrame", args):
  max_size = args.preprocess_max_size

  if "is_reduced" in dataset.columns and not args.override:
    print("  reduction already performed -- skipping.")
  else:
    print("  reducing large images... This may take a while.")

    items = [(r.full_path, max_size) for _, r in dataset.iterrows()]

    if args.preprocess_workers <= 1:
      paths_and_reduced = list(map(_reduce_image, items))
    else:
      with concurrent.futures.ThreadPoolExecutor(max_workers=args.preprocess_workers) as executor:
        paths_and_reduced = list(executor.map(_reduce_image, items))

    new_paths, is_reduced = list(zip(*paths_and_reduced))

    dataset["full_path"] = new_paths
    dataset["is_reduced"] = is_reduced

    dataset.to_csv(get_cleaned_dataset_path(args.data_dir), index=False)

  print(f"  reduced images (size>{max_size}px): {dataset.is_reduced.sum()} ({dataset.is_reduced.mean():.0%})")


def _reduce_image(args):
  image_path, max_size = args
  path, ext = os.path.splitext(image_path)
  output_path = path + "_scaled" + ext

  if os.path.exists(output_path):
    return output_path, True

  with Image.open(image_path) as img:
    W, H = img.size
    ratio = H / W

    if max(img.size) <= max_size:
      return image_path, False

    print(f"reducing {image_path} ({H}, {W})", flush=True)

    if W > H:
      W, H = int(max_size), int(max_size * ratio)
    else:
      W, H = int(max_size / ratio), int(max_size)

    img.convert("RGB").resize((W, H), resample=Image.Resampling.BICUBIC).save(output_path)

  return output_path, True
