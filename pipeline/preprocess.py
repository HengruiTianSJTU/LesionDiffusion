import os
import argparse
import json
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from nibabel.orientations import io_orientation, ornt_transform
from scipy.ndimage import zoom
from totalsegmentator.python_api import totalsegmentator
from ldm.data.Torchio_contrast_dataloader import totalseg_class
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from scipy.ndimage import binary_erosion, generate_binary_structure
from openai import OpenAI 

llm_url="https://api.openai.com/v1/chat/completions"
llm_model="gpt-3.5-turbo"

# Global description dictionary used to guide the LLM in generating a report.
description = {
    "enhancement status": [["enhanced ct,"], ["non-contrast ct,"]],
    "shape": [["round-like"], ["irregular"], ["irregular wall thickening", "irregular,wall thickening"], ["punctate", "nodular"], ["cystic"], ["luminal narrowing"], ["protrusion into the lumen"]],
    "density": [["hypodense lesion", "low density lesion", "hypoattenuating lesion", "low attenuation lesion"],
                ["isodense lesion", "isoattenuating lesion"],
                ["hyperdense lesion", "high density lesion", "hyperattenuating lesion", "high attenuation lesion"],
                ["mixed-density lesion", "mixed attenuation lesion"],
                ["hypoattenuating fluid-like lesion", "hypodense fluid-like lesion"],
                ["isodense soft tissue mass"],
                ["isodense soft tissue mass with peripheral low-density ground-glass opacity"],
                ["low-density ground-glass opacity"],
                ["ring enhancement"]],
    "density variations": [["homogeneous"], ["heterogeneous"]],
    "surface characteristics": [["well-defined margin"], ["clear serosal surface"], ["clear boundary with the renal parenchyma"],
                                ["ill-defined margin"], ["serosal surface irregularity"], ["unclear boundary with the renal parenchyma"],
                                ["poorly defined boundary with normal esophageal tissue"]],
    "relationship with adjacent organs": [["no close relationship with surrounding organs"],
                                          ["close relationship with adjacent organs"]],
    "specific features": [["presence of decreased density areas", "presence of decreased attenuation areas"],
                          ["presence of increased density areas", "presence of increased attenuation areas"],
                          ["presence of increased attenuation areas, presence of decreased attenuation areas"],
                          ["spiculated margins"],
                          ["retention of pancreatic fluid"],
                          ["stone", "calculus"]],
    "cavity": [["within the lumen of a hollow organ", "wall thickening"],
               ["within the parenchymal organ", "protruding from the parenchymal organ"]]
}


def standardize_image(target_orient, img_path):
    """
    Standardize NIfTI image orientation and voxel spacing.
    """
    img = nib.load(img_path)
    current_orient = io_orientation(img.affine)
    transform = ornt_transform(current_orient, target_orient)
    img = img.as_reoriented(transform)
    data = img.get_fdata()

    spacing = list(img.header.get_zooms())
    if data.ndim == 4:
        data = data[..., 0]
        spacing = spacing[:3]
    target_spacing = [1, 1, 1]
    zoom_factor = np.array(spacing) / np.array(target_spacing)
    order = 1 if data.min() < -10 else 0
    data = zoom(data, zoom=zoom_factor, order=order)

    new_img = nib.Nifti1Image(data, img.affine)
    new_img.header.set_zooms(target_spacing)
    nib.save(new_img, img_path)

    # Reload and re-save with SimpleITK for compatibility
    sitk_img = sitk.ReadImage(img_path)
    sitk.WriteImage(sitk_img, img_path)
    return new_img


def segment_image(img_path, original_img):
    """
    Perform segmentation using TotalSegmentator and adjust metadata so that the output
    (totalseg.nii.gz) retains the qoffset, srow, and pixdim fields from the original image.
    Returns a tuple (segmentation file path, segmentation image object).
    """
    output_path = os.path.join(os.path.dirname(img_path), "totalseg.nii.gz")
    try:
        seg_img = totalsegmentator(original_img, ml=True, quiet=True, v1_order=True)
        seg_data = seg_img.get_fdata().astype(np.uint8)
        seg_hdr = seg_img.header.copy()
        orig_hdr = original_img.header

        seg_hdr.set_qform(original_img.affine, code=int(orig_hdr['qform_code']))
        seg_hdr.set_sform(original_img.affine, code=int(orig_hdr['sform_code']))
        seg_hdr.set_zooms(orig_hdr.get_zooms()[:3])
        seg_hdr['pixdim'] = orig_hdr['pixdim']

        new_seg = nib.Nifti1Image(seg_data, original_img.affine, seg_hdr)
        nib.save(new_seg, output_path)
        return output_path, new_seg
    except Exception as e:
        print(f"Segmentation failed for {img_path}: {e}")
        return None, None


def compute_target_bbox(seg_data, organ_label):
    """
    Compute the bounding box size for the target organ in the segmentation data.
    Returns an array [size_x, size_y, size_z] representing the extents (in voxels).
    """
    coords = np.argwhere(seg_data == int(organ_label))
    if len(coords) == 0:
        raise ValueError(f"No voxels found for organ label {organ_label}.")
    min_coords = coords.min(axis=0)
    max_coords = coords.max(axis=0)
    bbox_size = (max_coords - min_coords) + 1  # +1 to include the boundaries
    return bbox_size


def generate_report(organ_label, target_bbox_size, attribute_dict, api_key):
    """
    Calls OpenAI API to generate a structured radiology report.
    The prompt includes the target organ's bounding box size, expected lesion attributes,
    and a predefined JSON format to ensure structured output.
    
    Returns:
        Tuple (report, bbox_size) where:
        - report: Generated structured radiology report as a dictionary.
        - bbox_size: Extracted lesion bounding box size as a NumPy array.
    """

    # Initialize OpenAI API client
    client = OpenAI(api_key=api_key, base_url=llm_url)

    # Convert target bounding box size to string format for prompt
    target_bbox_str = ", ".join(map(str, target_bbox_size.tolist()))
    
    # Construct prompt with strict JSON format
    prompt = (
        f"Generate a structured radiology report for a CT scan showing {attribute_dict['lesion type']} in a target organ {attribute_dict['organ type']}. "
        f"The target organ has a bounding box size of [[{target_bbox_str}]] voxels. "
        f"Please consider the following expected characteristics as guidelines: {json.dumps(description)}. "
        f"Additionally, the expected attributes for this case are: {json.dumps(attribute_dict)}. "
        f"Ensure that the lesion bounding box size([[X, Y, Z]] pixel) does not exceed the target organ's bounding box size. "
        f"Ensure that the output follows this JSON structure precisely:\n\n"
        f'{{'
        f'    {organ_label}: ['
        f'        "<describe shape>",'
        f'        "",'
        f'        {{'
        f'            "enhancement status": "<describe contrast enhancement>",'
        f'            "lesion location": {organ_label},'
        f'            "shape": "<describe shape>",'
        f'            "size": "<provide size in format [[X, Y, Z]] pixel>",'
        f'            "density": "<describe density>",'
        f'            "density variations": "<describe homogeneity>",'
        f'            "surface characteristics": "<describe lesion margin>",'
        f'            "relationship with adjacent organs": "<describe involvement>",'
        f'            "specific features": "<mention any distinguishing features>"'
        f'            "cavity": "<describe cavity>",'
        f'        }},'
        f'        [[[{target_bbox_str}]], [[X, Y, Z]]]'
        f'    ]'
        f'}}'
    )

    # Call OpenAI API to generate the report
    response = client.chat.completions.create(
        model=llm_model,
        messages=[
            {"role": "system", "content": "You are a radiology expert."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=300
    )

    # Parse JSON response
    try:
        raw_content = response.choices[0].message.content.strip()

        # Remove markdown-like code block formatting
        if raw_content.startswith("```json"):
            raw_content = raw_content[7:]  # Remove "```json\n"
        if raw_content.endswith("```"):
            raw_content = raw_content[:-3]  # Remove trailing "```"

        # Convert to Python dictionary
        gpt_report = json.loads(raw_content)
    except Exception as e:
        raise ValueError(f"Error parsing JSON from OpenAI response: {e}")

    # Ensure the report contains the expected organ label
    if organ_label not in gpt_report:
        raise ValueError(f"Missing organ label {organ_label} in generated report.")

    # Extract report details
    report_details = gpt_report[organ_label][2]

    # Override model-generated attributes with those provided in attribute_dict
    for key in ["enhancement status", "shape", "density", "density variations", "cavity",
                "surface characteristics", "relationship with adjacent organs", "specific features"]:
        if key in attribute_dict:
            report_details[key] = attribute_dict[key]

    # Extract lesion size information
    try:
        size_str = gpt_report[organ_label][3][1]
        size_values = size_str[0]
        # Ensure exactly three numerical values (X, Y, Z)
        if len(size_values) != 3:
            raise ValueError(f"Size format error: {size_str}")

    except Exception as e:
        print(f"Warning: Error parsing bbox size from report: {e}. Using fallback size {target_bbox_size.tolist()}")

    bbox_size = np.array(size_values)
    return gpt_report, bbox_size


def generate_random_bbox_from_mask(seg_data, organ_label, bbox_size, desired_cavity, organ_type):
    """
    Generate a random bounding box within seg_data where voxels equal organ_label.
    The selection of the bbox center is guided by the 'cavity' key in attribute_dict.
    
    For hollow organs (urinary bladder, stomach, colon, esophagus):
      - If cavity == "within the lumen of a hollow organ": select the bbox center from the eroded (inner) region.
      - If cavity == "wall thickening": select the bbox center from the boundary (mask minus eroded mask).
    
    For other solid organs:
      - If cavity == "within the parenchymal organ": ensure the entire bbox is inside the organ mask.
      - If cavity == "protruding from the parenchymal organ": only ensure the bbox center is inside the mask.
    
    Returns center as numpy arrays.
    """
    mask = (seg_data == int(organ_label))
    if not np.any(mask):
        raise ValueError(f"No voxels found for organ label {organ_label}.")
    hollow_organs = {"urinary bladder", "stomach", "colon", "esophagus"}
    organ_lower = organ_type.lower()
    if organ_lower in hollow_organs and desired_cavity == "within the lumen of a hollow organ":
        # Compute inner cavity using binary erosion (adjust iterations as needed)
        structure = generate_binary_structure(3, 1)
        eroded = binary_erosion(mask, structure=structure, iterations=5)
        indices = np.argwhere(eroded)
        if len(indices) == 0:
            indices = np.argwhere(mask)
        center = indices[np.random.randint(len(indices))]
    elif organ_lower in hollow_organs and desired_cavity == "wall thickening":
        # Compute boundary as mask minus eroded mask
        structure = generate_binary_structure(3, 1)
        eroded = binary_erosion(mask, structure=structure, iterations=5)
        boundary = mask & (~eroded)
        indices = np.argwhere(boundary)
        if len(indices) == 0:
            indices = np.argwhere(mask)
        center = indices[np.random.randint(len(indices))]
    elif organ_lower not in hollow_organs:
        # For solid organs
        if desired_cavity == "within the parenchymal organ":
            # Ensure the entire bbox region is inside the organ mask.
            indices = np.argwhere(mask)
            np.random.shuffle(indices)
            found = False
            for candidate in indices:
                half_size = bbox_size // 2
                min_corner = candidate - half_size
                max_corner = candidate + half_size
                valid = True
                for dim in range(3):
                    if min_corner[dim] < 0 or max_corner[dim] >= seg_data.shape[dim]:
                        valid = False
                        break
                if not valid:
                    continue
                bbox_region = mask[min_corner[0]:max_corner[0] + 1, min_corner[1]:max_corner[1] + 1, min_corner[2]:max_corner[2] + 1]
                if np.all(bbox_region):
                    center = candidate
                    found = True
                    break
            if not found:
                indices = np.argwhere(mask)
                center = indices[np.random.randint(len(indices))]
        elif desired_cavity == "protruding from the parenchymal organ":
            # Only ensure bbox center is inside the mask (and bbox is within image boundaries)
            indices = np.argwhere(mask)
            center = indices[np.random.randint(len(indices))]
        else:
            indices = np.argwhere(mask)
            center = indices[np.random.randint(len(indices))]
    else:
        indices = np.argwhere(mask)
        center = indices[np.random.randint(len(indices))]

    # Adjust center to ensure the bbox remains within image boundaries
    shape = np.array(seg_data.shape)
    half_size = bbox_size // 2
    for i in range(3):
        if center[i] - half_size[i] < 0:
            center[i] = half_size[i]
        if center[i] + half_size[i] >= shape[i]:
            center[i] = shape[i] - half_size[i] - 1
    return center


def create_bbox_mask(seg_img, bbox_center, bbox_size, folder):
    """
    Create a bbox mask using the same header as seg_img.
    The bbox region is set to 2 while the background remains 0.
    Returns the output file path of bbox.nii.gz.
    """
    seg_data = seg_img.get_fdata()
    bbox_data = np.zeros(seg_data.shape, dtype=np.uint8)
    half_size = bbox_size // 2
    min_corner = np.maximum(bbox_center - half_size, 0)
    max_corner = np.minimum(bbox_center + half_size, np.array(seg_data.shape))
    bbox_data[min_corner[0]:max_corner[0], min_corner[1]:max_corner[1], min_corner[2]:max_corner[2]] = 2

    bbox_path = os.path.join(folder, "bbox.nii.gz")
    bbox_img = nib.Nifti1Image(bbox_data, seg_img.affine, seg_img.header)
    nib.save(bbox_img, bbox_path)
    return bbox_path


def process_file(img_path, attribute_dict, api_key):
    """
    Complete processing of a single file:
      1. Image standardization.
      2. Segmentation.
      3. Compute the target organ's bbox size from segmentation.
      4. Call the OpenAI API to generate a report, passing the target bbox size and expected attributes.
      5. Parse the lesion bbox size from the report.
      6. Generate a lesion bbox within the target organ mask according to the provided 'cavity' guidelines.
      7. Create the bbox mask.
      8. Save the report as type.json.
    Returns a tuple: (img_path, seg_path, type_json_path, bbox_path)
    """
    folder = os.path.dirname(img_path)
    try:
        std_img = standardize_image(np.array([[0, -1], [1, -1], [2, 1]]), img_path)
        seg_path, seg_img = segment_image(img_path, std_img)
        if seg_img is None:
            return None

        # Extract organ type and lesion type from attribute_dict.
        if "organ type" not in attribute_dict or "lesion type" not in attribute_dict:
            raise ValueError(f"Attribute dictionary for {img_path} must include 'organ type' and 'lesion type'.")
        organ_type = attribute_dict["organ type"]

        # Determine organ_label using totalseg_class dictionary.
        organ_label = next((key for key, val in totalseg_class.items() if val == organ_type), None)
        if organ_label is None:
            raise ValueError(f"Organ {organ_type} not found in totalseg_class dictionary.")

        # Compute the target organ's bbox size.
        seg_data = seg_img.get_fdata()
        target_bbox_size = compute_target_bbox(seg_data, organ_label)

        # Generate the report and parse the lesion bbox size.
        report, bbox_size = generate_report(organ_label, target_bbox_size, attribute_dict, api_key)
        cavity = report[organ_label][2]['cavity']
        type_json_path = os.path.join(folder, "type.json")
        with open(type_json_path, "w") as f:
            json.dump(report, f, indent=0)

        # Generate a lesion bbox within the target organ mask using cavity guidance.
        bbox_center= generate_random_bbox_from_mask(seg_data, organ_label, bbox_size, cavity, organ_type)
        bbox_path = create_bbox_mask(seg_img, bbox_center, bbox_size, folder)

        return img_path, seg_path, type_json_path, bbox_path

    except Exception as e:
        print(f"Processing failed for {img_path}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Pipeline for inference image preprocessing")
    parser.add_argument("filelist", type=str, help="Path to preprocess_img_list.txt")
    parser.add_argument("exp_name", type=str, help="Experiment name for result tracking")
    parser.add_argument("attributes", type=str,
                        help="A path to an attributes.txt file (each line is a JSON dictionary including 'organ type' and 'lesion type')")
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY environment variable.")

    # Read image paths.
    with open(args.filelist, "r") as f:
        img_paths = [line.strip() for line in f if line.strip()]

    # Parse attributes parameter.
    attribute_list = []
    if os.path.exists(args.attributes):
        # Read each line and parse it as a JSON dictionary.
        with open(args.attributes, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        attr = json.loads(line)
                        attribute_list.append(attr)
                    except Exception as e:
                        print(f"Error parsing attribute line: {line}. Error: {e}")
                        raise
        if len(attribute_list) != len(img_paths):
            raise ValueError("Number of attribute entries does not match number of image paths.")
    else:
        raise ValueError(f"No such a path exsits: {e}")

    exp_dir = os.path.join(os.getcwd(), args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    img_list, seg_list, type_list, bbox_list = [], [], [], []

    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = {executor.submit(process_file, img, attribute, api_key): img
                   for img, attribute in zip(img_paths, attribute_list)}
        for future in  tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            result = future.result()
            if result:
                img_list.append(result[0])
                seg_list.append(result[1])
                type_list.append(result[2])
                bbox_list.append(result[3])

    for name, lst in zip(["img_list", "seg_list", "type_list", "bbox_list"],
                         [img_list, seg_list, type_list, bbox_list]):
        with open(os.path.join(exp_dir, f"{name}.txt"), "w") as f:
            f.write("\n".join(lst) + "\n")

    print(f"Processing completed! Results saved in {exp_dir}")


if __name__ == "__main__":
    main()
