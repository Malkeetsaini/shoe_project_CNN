import os
import pandas as pd

ROOT_DIR = "data/shoes"

data = []

for main_category in os.listdir(ROOT_DIR):

    main_path = os.path.join(ROOT_DIR, main_category)

    if not os.path.isdir(main_path):
        continue

    for subcategory in os.listdir(main_path):

        sub_path = os.path.join(main_path, subcategory)

        if not os.path.isdir(sub_path):
            continue

        for brand in os.listdir(sub_path):

            brand_path = os.path.join(sub_path, brand)

            if not os.path.isdir(brand_path):
                continue

            for image_name in os.listdir(brand_path):

                if image_name.lower().endswith(
                    (".jpg", ".jpeg", ".png")
                ):

                    image_path = os.path.join(
                        brand_path,
                        image_name
                    )

                    # Shoe type mapping
                    if main_category == "Boots":
                        shoe_type = "boot"

                    elif main_category == "Slippers":
                        shoe_type = "slipper"

                    else:
                        shoe_type = "formal"

                    data.append([
                        image_path,
                        main_category,
                        subcategory,
                        shoe_type
                    ])

df = pd.DataFrame(
    data,
    columns=[
        "image_path",
        "main_category",
        "subcategory",
        "shoe_type"
    ]
)

df = df.sample(frac=1).reset_index(drop=True)

train_size = int(0.8 * len(df))
val_size = int(0.1 * len(df))

train_df = df[:train_size]
val_df = df[train_size:train_size+val_size]
test_df = df[train_size+val_size:]

train_df.to_csv("data/train.csv", index=False)
val_df.to_csv("data/val.csv", index=False)
test_df.to_csv("data/test.csv", index=False)

print("CSV files created successfully!")
print("Total images:", len(df))