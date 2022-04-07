import torch
import model
import coremltools as ct
import argparse
from convert_state import convert_state_dict
import torchvision.transforms.functional as TF


def convert():
    parser = argparse.ArgumentParser(
        description="Convert Pytorch model to CoreML model")
    parser.add_argument("--model_file", type=str, required=True, help="Path to model checkpoint file")
    parser.add_argument("--out_file", type=str, required=True, help="Path to the output mlmodel file")

    args = parser.parse_args()

    # Tracing works, so we're good here
    example_img, example_mask, example_target = torch.rand(1, 3, 320, 320), torch.rand(1, 1, 320, 320), torch.rand(1, 3, 1000, 1000)
    polylayer = model.Deg4MobilePolyLayer()  # This is the polylayer compatible with Core ML
    net = model.TriSpaceRegNet(polynomial_order=4, spatial=True, is_train=False, polylayer=polylayer)
    checkpoint = torch.load(args.model_file, map_location=torch.device('cpu'))

    # Converts DP/DDP model state to regular model state
    new_state_dict = convert_state_dict(checkpoint['model_state_dict'])
    net.load_state_dict(new_state_dict)

    net.eval()  # https://github.com/pytorch/pytorch/issues/23999
    trace = torch.jit.trace(net, [example_img, example_mask])

    img_shape = ct.Shape(shape=(1, 3, 320, 320))  # RangeDim makes shape variable along dim
    mask_shape = ct.Shape(shape=(1, 1, 320, 320))
    # Can also use TensorType, works perfect
    target_shape = ct.Shape(shape=(1, 3, ct.RangeDim(1, 10000), ct.RangeDim(1, 10000)))  # RangeDim makes shape variable along dim
    img_in = ct.ImageType(name='image', shape=img_shape, scale=1.0/255.0)  # Using ImageType as this is the preferred type for image data
    mask_in = ct.ImageType(name='mask', shape=mask_shape, scale=1.0/255.0)
    model_from_torch = ct.convert(trace, inputs=[img_in, mask_in], debug=True)
    model_from_torch.save(args.out_file)

    ctnet = ct.models.model.MLModel(args.out_file, useCPUOnly=True,  compute_units=ct.ComputeUnit.CPU_ONLY)
    output = ctnet.predict({"image": TF.to_pil_image(example_img[0]),
                            "mask": TF.to_pil_image(example_mask[0])})


if __name__ == '__main__':
    convert()
