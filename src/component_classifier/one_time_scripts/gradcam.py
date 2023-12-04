import panel as pn
import torch
from IPython.display import display
from mirror.visualisations.core import GradCam
from PIL import Image
from torchvision.models import ResNet50_Weights

from component_classifier.data_utils import LABEL_COLS, get_metadata_df, load_model
from component_classifier.train_utils import batches_to_uint8

pn.extension()


def apply_gradcam(img, true_label, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = ResNet50_Weights.IMAGENET1K_V2.transforms()
    transformed_img = transform(img).unsqueeze(0).to(device)

    gradcam_model = GradCam(model, device=device)

    # sizes:
    # 0 torch.Size([112, 112])
    # 11 torch.Size([56, 56])
    # 24 torch.Size([28, 28])
    # 43 torch.Size([14, 14])
    # 52 torch.Size([7, 7])
    module = [m for m in model.modules() if isinstance(m, torch.nn.Conv2d)][52]

    def norm(x):
        return (x - x.min()) / (x.max() - x.min()).clamp(0, 1)

    match gradcam_model(input_image=transformed_img, layer=module, postprocessing=norm):
        case raw_gradcam, {"prediction": pred_label, "cam": raw_cam_mask}:
            ...

    gradcam_img = Image.fromarray(batches_to_uint8(raw_gradcam).squeeze().permute(1, 2, 0).cpu().numpy()).resize(img.size)
    cam_mask_overlay = Image.fromarray(batches_to_uint8(raw_cam_mask).cpu().numpy()).resize(img.size)
    masked_img = img.convert("RGB").convert("RGBA")
    masked_img.putalpha(cam_mask_overlay)

    return {
        "true_label": LABEL_COLS[true_label],
        "pred_label": LABEL_COLS[pred_label],
        "masked_img": masked_img,
        "gradcam_img": gradcam_img,
    }


if __name__ == "__main__":
    model = torch.load(
        r"E:\component-classifier\src\component_classifier\data\mlruns/0/78c580afbe994b158a0529d15ff70fdb/artifacts/model/data/model.pth"
    )
    # model = load_model("78c580afbe994b158a0529d15ff70fdb")

    dev_df = get_metadata_df().query('split == "dev"')

    cols = pn.Column()
    for _ in range(10):
        sample = dev_df.sample(1).squeeze()
        img = Image.open(sample.local_path).resize((256, 256))
        label = sample.label

        result = apply_gradcam(img, label, model)

        icon = "✅" if result["true_label"] == result["pred_label"] else "❌"
        cols += pn.Column(
            pn.pane.Str(f"{icon} True={result['true_label']}, Pred={result['pred_label']}"),
            pn.Row(
                pn.pane.image.PNG(img),
                pn.pane.image.PNG(result["masked_img"]),
                pn.pane.image.PNG(result["gradcam_img"]),
            ),
        )
    pn.serve(cols)
    display(cols)
