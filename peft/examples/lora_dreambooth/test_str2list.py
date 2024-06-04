import argparse

def str_to_list(arg):
    return [arg]

parser = argparse.ArgumentParser()

parser.add_argument(
    "--validation_prompt",
    type=str_to_list,
    default=[
        "masterpiece, best quality, (detailed face, perfect eyes, realistic eyes, perfect fingers), fantasy girl with long hair and autumn leaf clips, outdoors during sunset in a red velvet dress, (clear face), full body, amidst falling leaves.",
        "A photo of girl with green hair, detailed face, highres, RAW photo 8k uhd",
        "A photo of girl"
    ],
    help="A prompt that is used during validation to verify that the model is learning.",
)

if __name__ == "__main__":
    args = parser.parse_args()
    print(args.validation_prompt)
