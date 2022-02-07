import torch
import model as model
import torchvision.transforms.functional as TF
import torchvision.transforms as trans
from PIL import Image
import argparse
from convert_state import convert_state_dict


def infer():

    parser = argparse.ArgumentParser(
        description="Run image enhancement model on a single image")
    parser.add_argument("--img_path", type=str, required=True, help="Path to image to enhancement")
    parser.add_argument("--mask_path", type=str, required=True, help="Path to image to enhancement")
    parser.add_argument("--model_file", type=str, required=True, help="Path to model checkpoint file")
    parser.add_argument("--out_path", type=str, required=True, help="Path to write output image to")

    args = parser.parse_args()

    # This is the polylayer that is compatible with CoreML
    polylayer = model.Deg4MobilePolyLayer()
    net = model.TriSpaceRegNet(polynomial_order=4, spatial=True, is_train=False, polylayer=polylayer)
    net.eval()
    checkpoint = torch.load(args.model_file, map_location=torch.device('cpu'))

    # Converts DP/DDP model state to regular model state
    new_state_dict = convert_state_dict(checkpoint['model_state_dict'])
    net.load_state_dict(new_state_dict)

    # Load images, resize target image for input
    resizer = trans.Resize([320])
    cropper = trans.CenterCrop((320, 320))

    target_img, target_mask = Image.open(args.img_path), Image.open(args.mask_path).convert('L')
    img, mask = cropper(resizer(target_img)), cropper(resizer(target_mask))
    img_tensor, timg_tensor, mask_tensor, tmask_tensor = torch.unsqueeze(TF.to_tensor(img), 0), \
                                                         torch.unsqueeze(TF.to_tensor(target_img), 0), \
                                                         torch.unsqueeze(TF.to_tensor(mask), 0), \
                                                         torch.unsqueeze(TF.to_tensor(target_mask), 0)
    mask_tensor = (mask_tensor > 0).type(torch.FloatTensor)

    # Take output image, apply mask, turn background white like in app
    residual = net(img_tensor, mask_tensor, timg_tensor)
    out_img = net.generate_image(timg_tensor, residual)
    output_tensor = out_img * tmask_tensor + (1 - tmask_tensor)
    TF.to_pil_image(output_tensor[0]).save(args.out_path)


if __name__ == '__main__':
    infer()
