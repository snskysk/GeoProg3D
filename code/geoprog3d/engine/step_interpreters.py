import cv2
import os
import sys
import time
import torch
import openai
import functools
import numpy as np
import yaml
import json
import subprocess
from subprocess import PIPE
import io, tokenize
import augly.image as imaugs
from PIL import Image,ImageDraw,ImageFont,ImageFilter, ImageOps
from transformers import (ViltProcessor, ViltForQuestionAnswering, 
    OwlViTProcessor, OwlViTForObjectDetection,
    MaskFormerFeatureExtractor, MaskFormerForInstanceSegmentation,
    CLIPProcessor, CLIPModel, AutoProcessor, BlipForQuestionAnswering)
from vis_utils import html_embed_image, html_colored_span, vis_masks


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
CONFIG_FILE = os.path.join(PARENT_DIR, "config", "config.yml")


def parse_step(step_str,partial=False):
    tokens = list(tokenize.generate_tokens(io.StringIO(step_str).readline))
    output_var = tokens[0].string
    step_name = tokens[2].string
    parsed_result = dict(
        output_var=output_var,
        step_name=step_name)
    if partial:
        return parsed_result

    arg_tokens = [token for token in tokens[4:-3] if token.string not in [',','=']]
    num_tokens = len(arg_tokens) // 2
    args = dict()
    for i in range(num_tokens):
        args[arg_tokens[2*i].string] = arg_tokens[2*i+1].string
    parsed_result['args'] = args
    return parsed_result


def html_step_name(content):
    step_name = html_colored_span(content, 'red')
    return f'<b>{step_name}</b>'


def html_output(content):
    output = html_colored_span(content, 'green')
    return f'<b>{output}</b>'


def html_var_name(content):
    var_name = html_colored_span(content, 'blue')
    return f'<b>{var_name}</b>'


def html_arg_name(content):
    arg_name = html_colored_span(content, 'darkorange')
    return f'<b>{arg_name}</b>'

    
class EvalInterpreter():
    step_name = 'EVAL'

    def __init__(self):
        print(f'Registering {self.step_name} step')

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        output_var = parse_result['output_var']
        step_input = eval(parse_result['args']['expr'])
        assert(step_name==self.step_name)
        return step_input, output_var
    
    def html(self,eval_expression,step_input,step_output,output_var):
        eval_expression = eval_expression.replace('{','').replace('}','')
        step_name = html_step_name(self.step_name)
        var_name = html_var_name(output_var)
        output = html_output(step_output)
        expr = html_arg_name('expression')
        return f"""<div>{var_name}={step_name}({expr}="{eval_expression}")={step_name}({expr}="{step_input}")={output}</div>"""

    def execute(self,prog_step,inspect=False):
        step_input, output_var = self.parse(prog_step)
        prog_state = dict()
        for var_name,var_value in prog_step.state.items():
            if isinstance(var_value,str):
                if var_value in ['yes','no']:
                    prog_state[var_name] = var_value=='yes'
                elif var_value.isdecimal():
                    prog_state[var_name] = var_value
                else:
                    prog_state[var_name] = f"'{var_value}'"
            else:
                prog_state[var_name] = var_value
        
        eval_expression = step_input

        if 'xor' in step_input:
            step_input = step_input.replace('xor','!=')

        step_input = step_input.format(**prog_state)
        step_output = eval(step_input)
        prog_step.state[output_var] = step_output
        if inspect:
            html_str = self.html(eval_expression, step_input, step_output, output_var)
            return step_output, html_str

        return step_output


class ResultInterpreter():
    step_name = 'RESULT'

    def __init__(self):
        print(f'Registering {self.step_name} step')

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        output_var = parse_result['args']['var']
        assert(step_name==self.step_name)
        return output_var

    def html(self,output,output_var):
        step_name = html_step_name(self.step_name)
        output_var = html_var_name(output_var)
        if isinstance(output, Image.Image):
            output = html_embed_image(output,300)
        else:
            output = html_output(output)
            
        return f"""<div>{step_name} -> {output_var} -> {output}</div>"""

    def execute(self,prog_step,inspect=False):
        output_var = self.parse(prog_step)
        output = prog_step.state[output_var]
        if inspect:
            html_str = self.html(output,output_var)
            return output, html_str

        return output


class GetLandmarkSegInterpreter():
    step_name = 'GetLandmarkSeg'
    # ex SEG0=GetLandmarkSeg(query='The View')

    def __init__(self):
        print(f'Registering {self.step_name} step')

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        query_var = parse_result['args']['query']
        output_var = parse_result['output_var']
        assert(step_name==self.step_name)
        return query_var,output_var

    def html(self,query_var,output_var,one_thing_segment,pil_final_mask):
        step_name = html_step_name(self.step_name)
        output_var = html_var_name(output_var)
        query_arg = html_arg_name('query')
        pil_final_mask = html_embed_image(pil_final_mask)
        output = html_output(len(one_thing_segment))
        return f"""<div>{output_var}={step_name}({query_arg}={query_var})={output}px {pil_final_mask}</div>"""


    def execute(self,prog_step,inspect=False):
        query_var,output_var = self.parse(prog_step)
        with open(CONFIG_FILE, 'r') as yml:
            config = yaml.load(yml, Loader=yaml.SafeLoader)        
        json_open = open(config['geo_reference_json'], 'r')
        json_load = json.load(json_open)

        one_thing_segment = []
        for one_thing_info in json_load:
            one_thing_name = one_thing_info["object_name"]
            if one_thing_name == query_var[1:-1]:
                one_thing_segment = one_thing_info["segment"]
                break
        assert len(one_thing_segment) > 0
        for k, row in enumerate(one_thing_segment):
            x, y = row
            one_thing_segment[k] = [y, x]


        prog_step.state[output_var] = one_thing_segment
        prog_step.state[output_var+'_len'] = len(one_thing_segment)

        # Create PIL data for the mask
        M = np.array(one_thing_segment)
        height = config['topdown_view_parameter']['height']
        width = config['topdown_view_parameter']['width']
        image = np.zeros((height, width), dtype=np.uint8)
        mask = np.zeros_like(image, dtype=np.uint8)
        cv2.fillPoly(mask, [M], 255)
        pil_final_mask = Image.fromarray(mask)

        prog_step.state["comment"] = "none"
        prog_step.state[output_var+'_img'] = pil_final_mask        
        if inspect:
            html_str = self.html(query_var,output_var,one_thing_segment,pil_final_mask)
            return one_thing_segment, html_str

        return one_thing_segment

class SegAroundInterpreter():
    step_name = 'SegAround'
    # ex SEG3=SegAround(seg=SEG2,size=100) SEG3=SegAround(seg=SEG2)

    def __init__(self):
        print(f'Registering {self.step_name} step')

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        seg_var = parse_result['args']['seg']
        try:
            size_var = parse_result['args']['size']
        except:
            size_var = None

        output_var = parse_result['output_var']
        assert(step_name==self.step_name)
        return seg_var,size_var,output_var

    def html(self,seg_var,size_var,output_var,one_thing_segment,pil_final_mask):
        step_name = html_step_name(self.step_name)
        output_var = html_var_name(output_var)
        pil_final_mask = html_embed_image(pil_final_mask)
        output = html_output(len(one_thing_segment))
        seg_arg = html_arg_name('seg')
        size_arg = html_arg_name('size')

        return f"""<div>{output_var}={step_name}({seg_arg}={seg_var},{size_arg}={size_var})={output}pix {pil_final_mask}</div>"""

    def execute(self,prog_step,inspect=False):

        with open(CONFIG_FILE, 'r') as yml:
            config = yaml.load(yml, Loader=yaml.SafeLoader)        

        seg_var,size_var,output_var = self.parse(prog_step)
        list_M = prog_step.state[seg_var]        
        M = np.array(list_M)

        height = config['topdown_view_parameter']['height']
        width = config['topdown_view_parameter']['width']
        one_side_meter = config['topdown_view_parameter']['one_side_meter']
        
        image = np.zeros((height, width), dtype=np.uint8)

        if size_var == None:
            radius_size = int(height / 10)  # You can adjust the radius of the expansion mask by changing this value.
            if prog_step.state["comment"] == "CLOSEST":
                radius_size = 20
        else:
            radius_size = int((int(size_var) / one_side_meter)*height)


        white_img = np.full((height, width), 255, dtype=np.uint8)
        mask = np.zeros((height, width), dtype=bool)
        for x, y in list_M:
            mask[y, x] = True
        mask = np.ma.masked_array(white_img, mask=~mask)
        mask = mask.filled(0)

        # Expand mask
        kernel = np.ones((3, 3), np.uint8)
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)

        # Edge detection
        edges = cv2.Canny(dilated_mask, 100, 200)

        # Generate an extended mask
        expanded_mask = np.zeros_like(image, dtype=np.uint8)
        y_coords, x_coords = np.where(edges > 0)
        for y, x in zip(y_coords, x_coords):
            cv2.circle(expanded_mask, (x, y), radius_size, 255, -1)

        # Combining the original mask and the extended mask
        final_mask = cv2.bitwise_or(mask, expanded_mask)
        pil_final_mask = Image.fromarray(final_mask)
        # Extract the coordinates of the True parts from final_mask
        y_coords, x_coords = np.where(final_mask == 255)
        final_seg = [[int(x), int(y)] for x, y in zip(x_coords, y_coords)]

        assert len(final_seg) > 0
        prog_step.state[output_var] = final_seg
        prog_step.state[output_var+'_len'] = len(final_seg)
        prog_step.state[output_var+'_img'] = pil_final_mask        

        prog_step.state["comment"] = "none"
        if inspect:
            html_str = self.html(seg_var,size_var,output_var,final_seg,pil_final_mask)
            return final_seg, html_str

        return final_seg

class SegDirectionInterpreter():
    step_name = 'SegDirection'
    # ex SEG1=SegDirection(seg=SEG0,query='directly south')

    def __init__(self):
        print(f'Registering {self.step_name} step')

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        seg_var = parse_result['args']['seg']
        query_var = parse_result['args']['query']

        output_var = parse_result['output_var']
        assert(step_name==self.step_name)
        return seg_var,query_var,output_var

    def html(self,seg_var,query_var,output_var,one_thing_segment,pil_final_mask, rotated_angle):
        step_name = html_step_name(self.step_name)
        output_var = html_var_name(output_var)
        pil_final_mask = html_embed_image(pil_final_mask)
        output = html_output(len(one_thing_segment))
        seg_arg = html_arg_name('seg')
        query_arg = html_arg_name('query')

        return f"""<div>{output_var}={step_name}({seg_arg}={seg_var},{query_arg}={query_var})={output}pix {pil_final_mask} <br>(3DGS Topdown is a {rotated_angle}-degree rotation of Real Map.)</div>"""

    def execute(self,prog_step,inspect=False):
        from scipy import ndimage

        def rotate_matrix_preserve_values(matrix, angle_degrees):
            # Get the unique value of the original matrix
            unique_values = np.unique(matrix)
            
            # Rotate the matrix (using nearest neighbor interpolation)
            rotated_matrix = ndimage.rotate(matrix, angle_degrees, reshape=False, mode='constant', cval=0, order=0)
            
            # A function that rounds the matrix after rotation to its original value
            def round_to_original(x):
                if x == 0:  # Since 0 is a newly generated pixel, it is returned as is.
                    return 0
                # Returns the closest original value
                return unique_values[np.argmin(np.abs(unique_values - x))]
            
            # Apply the rounding function to the entire matrix
            rounded_rotated_matrix = np.vectorize(round_to_original)(rotated_matrix)
            
            return rounded_rotated_matrix.astype(matrix.dtype)

        def get_coordinates(array, value):
            # Use np.where() to obtain the coordinates of elements with specific values
            coordinates = np.argwhere(array == value)
            
            # Converting a NumPy array to a Python list
            coordinates_list = coordinates.tolist()
            
            return coordinates_list

        def calculate_new_M(map_array, M, direction, height, width):
            lower_direction = direction.lower()
            
            # Convert pixel list to set type
            M_set = set(map(tuple, M))
            
            # Calculating the boundary of M
            # x_coords, y_coords = np.array(M).T
            y_coords, x_coords = np.array(M).T
            x_min, x_max = x_coords.min(), x_coords.max()
            y_min, y_max = y_coords.min(), y_coords.max()
            
            # Calculate the center of M
            center_x = (x_min + x_max) // 2
            center_y = (y_min + y_max) // 2
            
            # Calculate a new set of pixels based on the direction
            x, y = np.meshgrid(np.arange(width), np.arange(height))
            mask = np.zeros((height, width), dtype=bool)
            
            if "north" in lower_direction:
                mask |= (y < y_min)
            if "south" in lower_direction:
                mask |= (y > y_max)
            if "east" in lower_direction:
                mask |= (x > x_max)
            if "west" in lower_direction:
                mask |= (x < x_min)
            
            if lower_direction.startswith("directly"):
                if "north" in lower_direction or "south" in lower_direction:
                    mask &= (x >= x_min) & (x <= x_max)
                if "west" in lower_direction or "east" in lower_direction:
                    mask &= (y >= y_min) & (y <= y_max)


            elif "northeast" in lower_direction:
                mask &= (x >= center_x) & (y < center_y)
            elif "northwest" in lower_direction:
                mask &= (x < center_x) & (y < center_y)
            elif "southeast" in lower_direction:
                mask &= (x >= center_x) & (y >= center_y)
            elif "southwest" in lower_direction:
                mask &= (x < center_x) & (y >= center_y)
            
            new_M = set(map(tuple, np.column_stack(np.where(mask))))
            
            # Exclude pixels contained in M
            new_M -= M_set
            
            return list(new_M)


        with open(CONFIG_FILE, 'r') as yml:
            config = yaml.load(yml, Loader=yaml.SafeLoader)        

        height = config['topdown_view_parameter']['height']
        width = config['topdown_view_parameter']['width']
        one_side_meter = config['topdown_view_parameter']['one_side_meter']
        rotated_angle = config['topdown_view_parameter']['rotated_angle']
        reverse_angle = 360 - rotated_angle

        seg_var,query_var,output_var = self.parse(prog_step)
        list_M = prog_step.state[seg_var]        
        M = np.array(list_M)
        for k, row in enumerate(M):
            x, y = row
            M[k] = [y, x]

        # Rotate the M to match the actual map.
        matrix = np.zeros((height, width))
        for pix_k, pix_xy in enumerate(M):        
            pix_x, pix_y = pix_xy
            # pix_y, pix_x = pix_xy
            matrix[pix_y, pix_x] = 1
        rotated_matrix = rotate_matrix_preserve_values(matrix, reverse_angle)
        y_coords, x_coords = np.where(rotated_matrix > 0.5)
        M = [[int(x), int(y)] for x, y in zip(x_coords, y_coords)]

        # Obtaining the direction from query
        direction = query_var[1:-1]        
        map_array = np.zeros((height, width), dtype=np.uint8)
        final_seg = calculate_new_M(map_array, M, direction, height, width)
        for k, row in enumerate(final_seg):
            x, y = row
            final_seg[k] = [y, x]

        # Reverse rotation
        matrix = np.zeros((height, width))
        for pix_k, pix_xy in enumerate(final_seg):        
            pix_x, pix_y = pix_xy
            # pix_y, pix_x = pix_xy
            matrix[pix_y, pix_x] = 1
        rotated_matrix = rotate_matrix_preserve_values(matrix, reverse_angle)
        y_coords, x_coords = np.where(rotated_matrix > 0.5)
        final_seg = [[int(x), int(y)] for x, y in zip(x_coords, y_coords)]

        # Create PIL data for the mask
        white_img = np.full((height, width), 255, dtype=np.uint8)
        mask = np.zeros((height, width), dtype=bool)
        for x, y in final_seg:
            mask[y, x] = True
        mask = np.ma.masked_array(white_img, mask=~mask)
        mask = mask.filled(0)
        pil_final_mask = Image.fromarray(mask)

        assert len(final_seg) > 0
        prog_step.state[output_var] = final_seg
        prog_step.state[output_var+'_len'] = len(final_seg)
        prog_step.state[output_var+'_img'] = pil_final_mask        

        prog_step.state["comment"] = "none"
        if inspect:
            html_str = self.html(seg_var,query_var,output_var,final_seg,pil_final_mask,rotated_angle)
            return final_seg, html_str

        return final_seg

class SegBetweenInterpreter():
    step_name = 'SegBetween'
    # ex SEG2=SegBetween(seg_first=SEG0,seg_second=SEG1)

    def __init__(self):
        print(f'Registering {self.step_name} step')

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        seg_first_var = parse_result['args']['seg_first']
        seg_second_var = parse_result['args']['seg_second']

        output_var = parse_result['output_var']
        assert(step_name==self.step_name)
        return seg_first_var,seg_second_var,output_var

    def html(self,seg_first_var,seg_second_var,output_var,one_thing_segment,pil_final_mask):
        step_name = html_step_name(self.step_name)
        output_var = html_var_name(output_var)
        pil_final_mask = html_embed_image(pil_final_mask)
        output = html_output(len(one_thing_segment))
        seg_first_arg = html_arg_name('seg_first')
        seg_second_arg = html_arg_name('seg_second')

        return f"""<div>{output_var}={step_name}({seg_first_arg}={seg_first_var},{seg_second_arg}={seg_second_var})={output}pix {pil_final_mask}</div>"""

    def execute(self,prog_step,inspect=False):

        with open(CONFIG_FILE, 'r') as yml:
            config = yaml.load(yml, Loader=yaml.SafeLoader)        

        seg_first_var,seg_second_var,output_var = self.parse(prog_step)
        list_M1 = prog_step.state[seg_first_var]        
        M1 = np.array(list_M1)
        for k, row in enumerate(M1):
            x, y = row
            M1[k] = [y, x]

        list_M2 = prog_step.state[seg_second_var]        
        M2 = np.array(list_M2)
        for k, row in enumerate(M2):
            x, y = row
            M2[k] = [y, x]

        height = config['topdown_view_parameter']['height']
        width = config['topdown_view_parameter']['width']
        one_side_meter = config['topdown_view_parameter']['one_side_meter']
        
        def calculate_intermediate_area(height, width, M1, M2):
            # Generate a binary image for the two input masks.
            M1_img = np.zeros((height, width), dtype=np.uint8)
            M2_img = np.zeros((height, width), dtype=np.uint8)

            # Fill in the pixels of M1 and M2 with white (255).
            for x, y in M1:
                M1_img[y, x] = 255
            for x, y in M2:
                M2_img[y, x] = 255

            # Preserve the original building area
            M1_original = M1_img.copy()
            M2_original = M2_img.copy()

            kernel = np.ones((3, 3), np.uint8)
            iteration = 0
            max_iterations = 500  # Maximum number of iterations to prevent infinite loops

            M1_dilated_touch_M2 = False
            M2_dilated_touch_M1 = False
            while iteration < max_iterations:
                # dilate M1
                M1_dilated = cv2.dilate(M1_img, kernel, iterations=1)
                # Check for overlap with the original M2 area
                if cv2.bitwise_and(M1_dilated, M2_original).any():
                    M1_dilated_touch_M2 = True
                M1_img = M1_dilated

                # dilate M2
                M2_dilated = cv2.dilate(M2_img, kernel, iterations=1)
                # Check for overlap with the original M1 area
                if cv2.bitwise_and(M2_dilated, M1_original).any():
                    M2_dilated_touch_M1 = True
                M2_img = M2_dilated

                if M1_dilated_touch_M2 and M2_dilated_touch_M1:
                    break

                iteration += 1

            # Extracting overlapping areas of dilation
            intersection = cv2.bitwise_and(M1_img, M2_img)

            # Exclude the original building area
            intermediate_area = cv2.bitwise_and(intersection, cv2.bitwise_not(cv2.bitwise_or(M1_original, M2_original)))

            # Returns the results as a list of coordinates
            result = np.where(intermediate_area > 0)
            # Convert a NumPy array to a regular Python list and cast it to an int type
            return [(int(x), int(y)) for x, y in zip(result[1], result[0])]

        final_seg = calculate_intermediate_area(height, width, M1, M2)
        for k, row in enumerate(final_seg):
            x, y = row
            final_seg[k] = [y, x]


        # Create PIL data for the mask
        white_img = np.full((height, width), 255, dtype=np.uint8)
        mask = np.zeros((height, width), dtype=bool)
        for x, y in final_seg:
            mask[y, x] = True
        mask = np.ma.masked_array(white_img, mask=~mask)
        mask = mask.filled(0)
        pil_final_mask = Image.fromarray(mask)

        assert len(final_seg) > 0
        prog_step.state[output_var] = final_seg
        prog_step.state[output_var+'_len'] = len(final_seg)
        prog_step.state[output_var+'_img'] = pil_final_mask        

        prog_step.state["comment"] = "none"
        if inspect:
            html_str = self.html(seg_first_var,seg_second_var,output_var,final_seg,pil_final_mask)
            return final_seg, html_str

        return final_seg

class GetSimilarAreasInterpreter():
    step_name = 'GetSimilarAreas'
    # ex AREAS0=GetSimilarAreas(query='park',area_filter_flag=True,area=SEG1)

    def __init__(self):
        print(f'Registering {self.step_name} step')

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        query_var = parse_result['args']['query']
        area_filter_flag_var = parse_result['args']['area_filter_flag']
        if area_filter_flag_var == "True":
            area_filter_flag_var = True
        else:
            area_filter_flag_var = False
        try:
            area_var = parse_result['args']['area']
        except:
            area_var = None
        output_var = parse_result['output_var']
        assert(step_name==self.step_name)
        return query_var,area_filter_flag_var,area_var,output_var

    def html(self,query_var,area_filter_flag_var,area_var,output_var,one_thing_segment,pil_final_mask):
        step_name = html_step_name(self.step_name)
        output_var = html_var_name(output_var)
        query_arg = html_arg_name('query')
        area_filter_flag_arg = html_arg_name('area_filter_flag')
        area_arg = html_arg_name('area')
        pil_final_mask = html_embed_image(pil_final_mask)
        output = html_output(len(one_thing_segment))
        return f"""<div>{output_var}={step_name}({query_arg}={query_var},{area_filter_flag_arg}={area_filter_flag_var},{area_arg}={area_var})={output}px {pil_final_mask}</div>"""

    def execute(self,prog_step,inspect=False):
        with open(CONFIG_FILE, 'r') as yml:
            config = yaml.load(yml, Loader=yaml.SafeLoader)        

        project_root = config["project_root"]
        similarity_count_json = f"{project_root}/geoprog3d/similarity_count.json"        

        query_var,area_filter_flag_var,area_var,output_var = self.parse(prog_step)
        if area_var == None:
            list_M = None
        else:
            list_M = prog_step.state[area_var]        
            for k, row in enumerate(list_M):
                x, y = row
                list_M[k] = [y, x]

        request_dict = {
            "query":query_var[1:-1],
            "area_filter_flag":area_filter_flag_var,
            "area":list_M
        }
        request_similar_area_json = config['request_similar_area_json']
        with open(request_similar_area_json, 'w') as f:
            json.dump(request_dict, f)

        process_finish_flag = config['process_finish_flag']

        if os.path.exists(process_finish_flag):
            os.remove(process_finish_flag)

        if not os.path.exists(similarity_count_json):
            # Tasks other than GRD
            command = "sh grd_processor.sh"
            proc = subprocess.run(command, shell=True, stdout=PIPE, stderr=PIPE, text=True)

            height = config['topdown_view_parameter']['height']
            width = config['topdown_view_parameter']['width']

            with open(config['topdown_seglist_json'], 'r') as f:
                loaded_data = json.load(f)
            one_thing_segment = loaded_data["topdown_seg_list"]

            for k, row in enumerate(one_thing_segment):
                x, y = row
                one_thing_segment[k] = [x, height - y]

            assert len(one_thing_segment) > 0
            prog_step.state[output_var] = one_thing_segment
            prog_step.state[output_var+'_len'] = len(one_thing_segment)

            # Create PIL data for the mask
            white_img = np.full((height, width), 255, dtype=np.uint8)
            mask = np.zeros((height, width), dtype=bool)
            for x, y in one_thing_segment:
                mask[y, x] = True
            mask = np.ma.masked_array(white_img, mask=~mask)
            mask = mask.filled(0)
            pil_final_mask = Image.fromarray(mask)
            prog_step.state[output_var+'_img'] = pil_final_mask        

            prog_step.state["comment"] = "none"
            if inspect:
                html_str = self.html(query_var,area_filter_flag_var,area_var,output_var,one_thing_segment,pil_final_mask)
                return one_thing_segment, html_str
            return one_thing_segment


        else:
            # Obtaining the number of executions
            with open(similarity_count_json, 'r') as f:
                loaded_data = json.load(f)
            requirement_count = loaded_data["requirement_count"]

            # Execution
            if requirement_count == 0:
                os.remove(similarity_count_json)
                if inspect:
                    return None, None
                else:
                    return None
            elif requirement_count > 1:
                # Calculation for grounding open vocabulary objects in the middle step of a query
                command = "sh grd_processor.sh"
                proc = subprocess.run(command, shell=True, stdout=PIPE, stderr=PIPE, text=True)

                # Update remaining processing times
                res_dict = {"requirement_count":requirement_count - 1}
                with open(similarity_count_json, 'w') as f:
                    json.dump(res_dict, f)

                height = config['topdown_view_parameter']['height']
                width = config['topdown_view_parameter']['width']

                with open(config['topdown_seglist_json'], 'r') as f:
                    loaded_data = json.load(f)
                one_thing_segment = loaded_data["topdown_seg_list"]

                for k, row in enumerate(one_thing_segment):
                    x, y = row
                    one_thing_segment[k] = [x, height - y]


                assert len(one_thing_segment) > 0
                prog_step.state[output_var] = one_thing_segment
                prog_step.state[output_var+'_len'] = len(one_thing_segment)


                # Create PIL data for the mask
                white_img = np.full((height, width), 255, dtype=np.uint8)
                mask = np.zeros((height, width), dtype=bool)
                for x, y in one_thing_segment:
                    mask[y, x] = True
                mask = np.ma.masked_array(white_img, mask=~mask)
                mask = mask.filled(0)
                pil_final_mask = Image.fromarray(mask)
                prog_step.state[output_var+'_img'] = pil_final_mask        

                prog_step.state["comment"] = "none"
                if inspect:
                    html_str = self.html(query_var,area_filter_flag_var,area_var,output_var,one_thing_segment,pil_final_mask)
                    return one_thing_segment, html_str
                return one_thing_segment

            else requirement_count == 1:
                # Calculation for final grounding
                command = "sh grd_processor_end.sh"
                proc = subprocess.run(command, shell=True, stdout=PIPE, stderr=PIPE, text=True)
                os.remove(SIMILARITY_COUNT_JSON)
                if inspect:
                    return None, None
                else:
                    return None


class GetClusterInterpreter():
    step_name = 'GetCluster'
    # ex SEG2=GetCluster(seg=AREAS0,cluster_type='BIGGEST')
    # ex ANSWER0=GetCluster(seg=AREAS0,cluster_type='CLOSEST',from=SEG0)

    def __init__(self):
        print(f'Registering {self.step_name} step')

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        seg_var = parse_result['args']['seg']# ex AREAS0
        cluster_type_var = parse_result['args']['cluster_type']# ex AREAS0
        try:
            from_var = parse_result['args']['from']# ex SEG0
        except:
            from_var = None

        output_var = parse_result['output_var']
        assert(step_name==self.step_name)
        return seg_var,cluster_type_var,from_var,output_var

    def html(self,seg_var,cluster_type_var,from_var,output_var,one_thing_segment,pil_final_mask):
        step_name = html_step_name(self.step_name)
        output_var = html_var_name(output_var)
        pil_final_mask = html_embed_image(pil_final_mask)
        output = html_output(len(one_thing_segment))
        seg_arg = html_arg_name('seg')
        cluster_type_arg = html_arg_name('cluster_type')
        from_arg = html_arg_name('from')

        return f"""<div>{output_var}={step_name}({seg_arg}={seg_var},{cluster_type_arg}={cluster_type_var},{from_arg}={from_var})={output}pix {pil_final_mask}</div>"""

    def execute(self,prog_step,inspect=False):

        import numpy as np
        from sklearn.cluster import KMeans
        from scipy.spatial.distance import cdist
        from scipy.spatial import ConvexHull
        from shapely.geometry import Point, Polygon

        from scipy import ndimage
        from sklearn.cluster import DBSCAN

        def refine_mask(M, image_shape, n_clusters=5, cluster_type="BIGGEST", comparison_M=None, cluster_points_limit=30):
            """
            A function that improves the mask and returns the clustered mask and the new mask
            
            :param M: Original mask coordinates [[x1,y1], [x2,y2], ..., [xn,yn]]
            :param image_shape: Image size (height, width)
            :param n_clusters: Number of clusters
            :param cluster_type: "BIGGEST" or "CLOSEST"
            :param comparison_M: Comparative mask coordinates (used when cluster_type=“CLOSEST”)
            :return: (cluster_M_list, new_M)
            """
            M = np.array(M)
            
            kmeans = KMeans(n_clusters=n_clusters)
            kmeans.fit(M)
            
            cluster_M_list = []
            new_M = []
            
            for cluster in range(n_clusters):
                cluster_points = M[kmeans.labels_ == cluster]
                
                if len(cluster_points) < cluster_points_limit:  # Ignore clusters that are too small.
                    continue
                
                # Calculating the center and standard deviation of the cluster
                center = np.mean(cluster_points, axis=0)
                std_dev = np.std(cluster_points, axis=0)
                
                # Only the points within 3 standard deviations of the center are retained.
                distances = cdist([center], cluster_points)[0]
                valid_points = cluster_points[distances <= 3 * np.mean(std_dev)]
                
                # Convex hull created using valid points
                hull = ConvexHull(valid_points)
                
                # Add all the pixels in the convex hull to the mask
                x_min, y_min = np.min(valid_points, axis=0)
                x_max, y_max = np.max(valid_points, axis=0)
                cluster_M = []
                for x in range(int(x_min), int(x_max) + 1):
                    for y in range(int(y_min), int(y_max) + 1):
                        if _point_in_hull([x, y], valid_points[hull.vertices]):
                            cluster_M.append([x, y])
                
                cluster_M_list.append(cluster_M)
                new_M.extend(cluster_M)
            
            if cluster_type == "CLOSEST" and comparison_M is not None:
                comparison_center = np.mean(comparison_M, axis=0)
                cluster_centers = [np.mean(cluster, axis=0) for cluster in cluster_M_list]
                distances = cdist([comparison_center], cluster_centers)[0]
                closest_cluster_index = np.argmin(distances)
                
                cluster_M_list = [cluster_M_list[closest_cluster_index]]
                new_M = cluster_M_list[0]
            
            return cluster_M_list, new_M

        def _point_in_hull(point, hull):
            """Check if the point is inside the convex hull"""
            return Point(point).within(Polygon(hull))


        def process_mask(M, cluster_type="BIGGEST", comparison_M=None, min_pixels=100,height=2048,width=2048):
            mask = np.zeros((height, width), dtype=np.uint8)
            
            # Set mask coordinates to 1
            for x, y in M:
                mask[y, x] = 1
            
            # Noise removal and filling
            mask = ndimage.binary_opening(mask, structure=np.ones((3,3)))
            mask = ndimage.binary_closing(mask, structure=np.ones((5,5)))
            
            # Labeling of concatenated components
            labeled, num_features = ndimage.label(mask)
            
            # clustering
            coords = np.column_stack(np.where(mask > 0))
            db = DBSCAN(eps=5, min_samples=5).fit(coords)
            labels = db.labels_
            
            # Generate a mask for each cluster
            cluster_M_list = []
            for label in set(labels):
                if label == -1:  # Ignore the noise
                    continue
                cluster_coords = coords[labels == label]
                cluster_M = cluster_coords[:, [1, 0]].tolist()
                cluster_M_list.append(cluster_M)
            
            # Obtain comparison_M as a set type (for fast searching)
            comparison_set = set(map(tuple, comparison_M)) if comparison_M else set()
            
            # Only clusters that do not overlap with comparison_M are retained.
            non_overlapping_clusters = [
                cluster for cluster in cluster_M_list
                if not any(tuple(coord) in comparison_set for coord in cluster)
            ]
            
            if not non_overlapping_clusters:
                return [], []  # If there are no clusters that do not overlap
            
            if cluster_type == "BIGGEST":
                # Select the largest non-overlapping cluster
                selected_clusters = [max(non_overlapping_clusters, key=len)]
            elif cluster_type == "CLOSEST" and comparison_M is not None:
                # Calculating the center of comparison_M
                comp_center = np.mean(comparison_M, axis=0)
                
                # Calculate the distance between the center of each non-overlapping cluster and the center of comparison_M
                # However, only clusters with a minimum number of pixels or more are targeted.
                valid_clusters = [cluster for cluster in non_overlapping_clusters if len(cluster) >= min_pixels]
                
                if not valid_clusters:
                    return [], []  # If there are no valid clusters
                
                distances = [np.linalg.norm(np.mean(cluster, axis=0) - comp_center) for cluster in valid_clusters]
                
                # Select the nearest non-overlapping and valid cluster
                closest_cluster = valid_clusters[np.argmin(distances)]
                selected_clusters = [closest_cluster]
            else:
                selected_clusters = non_overlapping_clusters
            
            new_M = []
            for cluster in selected_clusters:
                new_M.extend(cluster)
            
            return selected_clusters, new_M


        with open(CONFIG_FILE, 'r') as yml:
            config = yaml.load(yml, Loader=yaml.SafeLoader)        

        seg_var,cluster_type_var,from_var,output_var = self.parse(prog_step)
        list_M = prog_step.state[seg_var]        
        if from_var == None:
            list_comparison_M = None
        else:        
            list_comparison_M = prog_step.state[from_var]        
        cluster_type = cluster_type_var[1:-1]# 'BIGGEST' to BIGGEST
        height = config['topdown_view_parameter']['height']
        width = config['topdown_view_parameter']['width']
        one_side_meter = config['topdown_view_parameter']['one_side_meter']

        cluster_M_list, new_M = process_mask(list_M, cluster_type=cluster_type, comparison_M=list_comparison_M, min_pixels=300,height=height,width=width)
        final_seg = new_M


        white_img = np.full((height, width), 255, dtype=np.uint8)
        mask = np.zeros((height, width), dtype=bool)
        for x, y in final_seg:
            mask[y, x] = True
        mask = np.ma.masked_array(white_img, mask=~mask)
        mask = mask.filled(0)
        pil_final_mask = Image.fromarray(mask)

        assert len(final_seg) > 0
        prog_step.state[output_var] = final_seg
        prog_step.state[output_var+'_len'] = len(final_seg)
        prog_step.state[output_var+'_img'] = pil_final_mask        

        prog_step.state["cluster_M_list"] = cluster_M_list

        prog_step.state["comment"] = "none"
        if cluster_type == "CLOSEST":
            prog_step.state["comment"] = "CLOSEST"
        
        if inspect:
            html_str = self.html(seg_var,cluster_type_var,from_var,output_var,final_seg,pil_final_mask)
            return final_seg, html_str

        return final_seg


class GroundingResultInterpreter():
    step_name = 'RESULT'
    # ex FINAL_RESULT=RESULT(var=ANSWER0)

    def __init__(self):
        print(f'Registering {self.step_name} step')

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        seg_var = parse_result['args']['var']
        output_var = parse_result['output_var']
        assert(step_name==self.step_name)
        return seg_var,output_var

    def html(self,output_var,view_img,heatmap_img):
        step_name = html_step_name(self.step_name)
        output_var = html_var_name(output_var)
        view_img = html_embed_image(view_img)
        heatmap_img = html_embed_image(heatmap_img)

        return f"""<div>{output_var}->{step_name}->view:{view_img} heatmap:{heatmap_img}</div>"""

    def execute(self,prog_step,inspect=False):

        with open(CONFIG_FILE, 'r') as yml:
            config = yaml.load(yml, Loader=yaml.SafeLoader)        
        height = config['topdown_view_parameter']['height']
        width = config['topdown_view_parameter']['width']
        project_root = config["project_root"]
        scene_name = config["scene_info"]["scene_name"]
        parent_path = "{}LoG/renders4visprog/".format(project_root)
        scene_path = "{}{}/".format(parent_path, scene_name)

        heatmap_info_json_path = "{}similarity/heatmap_list.json".format(scene_path)
        with open(heatmap_info_json_path, 'r') as f:
            loaded_data = json.load(f)
        all_heatmap_path_list = loaded_data["heatmap_path_list"]

        arbitrary_save_dir = "{}arbitrary_dir/".format(scene_path)
        arbitrary_img_path_json = "{}arbitrary_img_path.json".format(arbitrary_save_dir)
        arbitrary_img_dir = "{}images/".format(arbitrary_save_dir)
        with open(arbitrary_img_path_json, 'r') as f:
            loaded_data = json.load(f)
        arbitrary_view_image_list = loaded_data["arbitrary_img_path_list"]
        a_and_b_count_all_view = loaded_data["a_and_b_count_all_view"]
        a_and_b_count_max = max(a_and_b_count_all_view)
        a_and_b_count_max_index = a_and_b_count_all_view.index(a_and_b_count_max)

        seg_var,output_var = self.parse(prog_step)

        view_img = arbitrary_view_image_list[a_and_b_count_max_index]
        heatmap_img = all_heatmap_path_list[a_and_b_count_max_index]

        view_img = Image.open(view_img)
        heatmap_img = Image.open(heatmap_img)

        if inspect:
            html_str = self.html(output_var,view_img,heatmap_img)
            return "complete", html_str

        return "complete"

class MeasureDistInterpreter():
    step_name = 'MeasureDist'
    # ex ANSWER0=MeasureDist(from=SEG3, to=SEG2)

    def __init__(self):
        print(f'Registering {self.step_name} step')

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        from_var = parse_result['args']['from']
        to_var = parse_result['args']['to']

        output_var = parse_result['output_var']
        assert(step_name==self.step_name)
        return from_var,to_var,output_var

    def html(self,from_var,to_var,output_var,distance_meter,pil_res_img,one_side_meter):
        step_name = html_step_name(self.step_name)
        output_var = html_var_name(output_var)
        output = html_output(distance_meter)
        pil_res = html_embed_image(pil_res_img)
        from_arg = html_arg_name('from')
        to_arg = html_arg_name('to')

        return f"""<div>{output_var}={step_name}({from_arg}={from_var},{to_arg}={to_var})={distance_meter}meter {pil_res}<br>(Each side of the image is {one_side_meter} meters long)</div>"""

    def execute(self,prog_step,inspect=False):


        def calculate_distance_and_visualize(M1, M2, height, width, one_side_meter, dist_type="center to center"):
            def pixel_to_meter(pixel):
                return pixel * (one_side_meter / max(height, width))

            img = Image.new('RGB', (width, height), color='black')
            draw = ImageDraw.Draw(img)

            # Draw M1 & M2 in white
            for building in [M1, M2]:
                building = np.array(building)
                draw.polygon(list(map(tuple, building)), fill='white', outline='white')

            if dist_type == "center to center":
                center1 = np.mean(M1, axis=0)
                center2 = np.mean(M2, axis=0)
                distance_pixels = np.linalg.norm(center1 - center2)
                distance_meters = pixel_to_meter(distance_pixels)
                
                # Connect the center points with a red line.
                draw.line([tuple(center1), tuple(center2)], fill='yellow', width=10)
                
            elif dist_type == "edge to edge":
                distances = cdist(M1, M2)
                min_distance_pixels = np.min(distances)
                distance_meters = pixel_to_meter(min_distance_pixels)
                
                # Find the two closest points
                idx1, idx2 = np.unravel_index(np.argmin(distances), distances.shape)
                closest_point1 = tuple(M1[idx1])
                closest_point2 = tuple(M2[idx2])
                
                # Connect the two closest points with a red line.
                draw.line([closest_point1, closest_point2], fill='yellow', width=10)
            
            else:
                raise ValueError("Invalid dist_type. Use 'center to center' or 'edge to edge'.")

            return distance_meters, img

        with open(CONFIG_FILE, 'r') as yml:
            config = yaml.load(yml, Loader=yaml.SafeLoader)        

        from_var,to_var,output_var = self.parse(prog_step)
        list_M1 = prog_step.state[from_var]        
        M1 = np.array(list_M1)
        # for k, row in enumerate(M1):
        #     x, y = row
        #     M1[k] = [y, x]

        list_M2 = prog_step.state[to_var]        
        M2 = np.array(list_M2)
        # for k, row in enumerate(M2):
        #     x, y = row
        #     M2[k] = [y, x]

        height = config['topdown_view_parameter']['height']
        width = config['topdown_view_parameter']['width']
        one_side_meter = config['topdown_view_parameter']['one_side_meter']
        

        # Calculates the distance between centers and visualizes it
        distance_meter, pil_res_img = calculate_distance_and_visualize(M1, M2, height, width, one_side_meter, "center to center")
        distance_meter = round(distance_meter, 2)


        prog_step.state[output_var] = distance_meter
        prog_step.state[output_var + "_img"] = pil_res_img
        prog_step.state["comment"] = "none"
        if inspect:
            html_str = self.html(from_var,to_var,output_var,distance_meter,pil_res_img,one_side_meter)
            return distance_meter, html_str

        return distance_meter

class EstimationResultInterpreter():
    step_name = 'RESULT'
    # ex FINAL_RESULT=RESULT(var=ANSWER0)

    def __init__(self):
        print(f'Registering {self.step_name} step')

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        dist_var = parse_result['args']['var']
        output_var = parse_result['output_var']
        assert(step_name==self.step_name)
        return dist_var,output_var

    def html(self,output_var,distance_meter):
        step_name = html_step_name(self.step_name)
        output_var = html_var_name(output_var)
        output = html_output(distance_meter)

        return f"""<div>{output_var}->{step_name}->{distance_meter} meters</div>"""

    def execute(self,prog_step,inspect=False):

        dist_var,output_var = self.parse(prog_step)
        distance_meter = prog_step.state[dist_var]

        if inspect:
            html_str = self.html(output_var,distance_meter)
            return distance_meter, html_str

        return distance_meter

class ObjCountingInterpreter():
    step_name = 'ObjCounting'
    # ex ANSWER0=ObjCounting(query='skyscrapers',area_filter_flag=True,area=SEG1)

    def __init__(self):
        print(f'Registering {self.step_name} step')

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        query_var = parse_result['args']['query']
        area_filter_flag_var = parse_result['args']['area_filter_flag']
        if area_filter_flag_var == "True":
            area_filter_flag_var = True
        else:
            area_filter_flag_var = False
        try:
            area_var = parse_result['args']['area']
        except:
            area_var = None
        output_var = parse_result['output_var']
        assert(step_name==self.step_name)
        return query_var,area_filter_flag_var,area_var,output_var

    def html(self,query_var,area_filter_flag_var,area_var,output_var,count_max_arbitrary_view,count_max_arbitrary_result,count_max_value):
        step_name = html_step_name(self.step_name)
        output_var = html_var_name(output_var)
        query_arg = html_arg_name('query')
        area_filter_flag_arg = html_arg_name('area_filter_flag')
        area_arg = html_arg_name('area')
        count_max_arbitrary_view = html_embed_image(count_max_arbitrary_view)
        count_max_arbitrary_result = html_embed_image(count_max_arbitrary_result,size=140)
        output = html_output(count_max_value)

        return f"""<div>{output_var}={step_name}({query_arg}={query_var},{area_filter_flag_arg}={area_filter_flag_var},{area_arg}={area_var})={output} s -- principal view:{count_max_arbitrary_view} res:{count_max_arbitrary_result}</div>"""

    def execute(self,prog_step,inspect=False):
        with open(CONFIG_FILE, 'r') as yml:
            config = yaml.load(yml, Loader=yaml.SafeLoader)        
        
        query_var,area_filter_flag_var,area_var,output_var = self.parse(prog_step)

        if area_var == None:
            list_M = None
        else:
            list_M = prog_step.state[area_var]        
            for k, row in enumerate(list_M):
                x, y = row
                list_M[k] = [y, x]

        request_dict = {
            "query":query_var[1:-1],
            "area_filter_flag":area_filter_flag_var,
            "area":list_M
        }
        request_similar_area_json = config['request_similar_area_json']
        with open(request_similar_area_json, 'w') as f:
            json.dump(request_dict, f)

        process_finish_flag = config['process_finish_flag']
        if os.path.exists(process_finish_flag):
            os.remove(process_finish_flag)

        # flag
        counting_wake_flag_path = config['counting_wake_flag_path']
        with open(counting_wake_flag_path, mode="w") as f:
            f.write("flag")

        command = "sh cnt_processor.sh"
        proc = subprocess.run(command, shell=True, stdout=PIPE, stderr=PIPE, text=True)

        project_root = config["project_root"]
        scene_name = config["scene_info"]["scene_name"]
        parent_path = "{}LoG/renders4visprog/".format(project_root)
        scene_path = "{}{}/".format(parent_path, scene_name)
        counting_save_dir = "{}counting_dir/".format(scene_path)
        counting_img_path_json = "{}counting_img_path.json".format(counting_save_dir)
        with open(counting_img_path_json, 'r') as f:
            counting_res_dict = json.load(f)

        count_list = counting_res_dict["count_list"]
        arbitrary_view_image_list = counting_res_dict["arbitrary_view_image_list"]
        counting_result_image_list = counting_res_dict["counting_result_image_list"]
        count_max_value = max(count_list)
        count_max_index = count_list.index(count_max_value)
        count_max_arbitrary_view_path = arbitrary_view_image_list[count_max_index]
        count_max_arbitrary_result_path = counting_result_image_list[count_max_index]

        count_max_arbitrary_view = Image.open(count_max_arbitrary_view_path)
        count_max_arbitrary_result = Image.open(count_max_arbitrary_result_path)
        
        prog_step.state[output_var] = count_max_value
        prog_step.state["comment"] = "none"
        prog_step.state["numerous_view_path"] = count_max_arbitrary_result_path
        if inspect:
            html_str = self.html(query_var,area_filter_flag_var,area_var,output_var,count_max_arbitrary_view,count_max_arbitrary_result,count_max_value)
            return count_max_value, html_str

        return count_max_value

class GetAreaHeightInterpreter():
    step_name = 'GetAreaHeight'
    # ex ANSWER0=GetAreaHeight(seg=SEG4)

    def __init__(self):
        print(f'Registering {self.step_name} step')

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        try:
            seg_var = parse_result['args']['seg']
        except:
            seg_var = None
        output_var = parse_result['output_var']
        assert(step_name==self.step_name)
        return seg_var,output_var

    def html(self,seg_var,output_var,res_img,res_height):
        step_name = html_step_name(self.step_name)
        output_var = html_var_name(output_var)
        seg_arg = html_arg_name('seg')
        res_img = html_embed_image(res_img,size=140)
        output = html_output(res_height)

        return f"""<div>{output_var}={step_name}({seg_arg}={seg_var})={output} meter -- res:{res_img}</div>"""

    def execute(self,prog_step,inspect=False):
        with open(CONFIG_FILE, 'r') as yml:
            config = yaml.load(yml, Loader=yaml.SafeLoader)        
        
        seg_var,output_var = self.parse(prog_step)

        if seg_var == None:
            list_M = None
        else:
            list_M = prog_step.state[seg_var]        
            for k, row in enumerate(list_M):
                x, y = row
                list_M[k] = [y, x]

        request_dict = {
            "query":"none",
            "area_filter_flag":True,
            "area":list_M
        }
        request_similar_area_json = config['request_similar_area_json']
        with open(request_similar_area_json, 'w') as f:
            json.dump(request_dict, f)

        process_finish_flag = config['process_finish_flag']
        if os.path.exists(process_finish_flag):
            os.remove(process_finish_flag)

        # flag
        height_wake_flag_path = config['height_wake_flag_path']
        with open(height_wake_flag_path, mode="w") as f:
            f.write("flag")

        command = "sh mes-h_processor.sh"
        proc = subprocess.run(command, shell=True, stdout=PIPE, stderr=PIPE, text=True)

        project_root = config["project_root"]
        scene_name = config["scene_info"]["scene_name"]
        parent_path = "{}LoG/renders4visprog/".format(project_root)
        scene_path = "{}{}/".format(parent_path, scene_name)
        height_save_dir = "{}height_dir/".format(scene_path)
        height_json = "{}height.json".format(height_save_dir)
        with open(height_json, 'r') as f:
            height_res_dict = json.load(f)

        obj_height = height_res_dict["obj_height"]
        height_fig_path = height_res_dict["height_fig_path"]
        res_img = Image.open(height_fig_path)
        
        prog_step.state[output_var] = obj_height
        prog_step.state["comment"] = "none"
        if inspect:
            html_str = self.html(seg_var,output_var,res_img,obj_height)
            return obj_height, html_str

        return obj_height


class CountingResultInterpreter():
    step_name = 'RESULT'
    # ex FINAL_RESULT=RESULT(var=ANSWER0)

    def __init__(self):
        print(f'Registering {self.step_name} step')

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        count_var = parse_result['args']['var']
        output_var = parse_result['output_var']
        assert(step_name==self.step_name)
        return count_var,output_var

    def html(self,output_var,max_count_value,count_max_arbitrary_result):
        step_name = html_step_name(self.step_name)
        output_var = html_var_name(output_var)
        output = html_output(max_count_value)
        count_max_arbitrary_result = html_embed_image(count_max_arbitrary_result,size=250)

        return f"""<div>{output_var}->{step_name}->{output} s -- {count_max_arbitrary_result}</div>"""

    def execute(self,prog_step,inspect=False):
        count_var,output_var = self.parse(prog_step)
        max_count_value = prog_step.state[count_var]
        count_max_arbitrary_result_path = prog_step.state["numerous_view_path"]
        count_max_arbitrary_result = Image.open(count_max_arbitrary_result_path)

        if inspect:
            html_str = self.html(output_var,max_count_value,count_max_arbitrary_result)
            return max_count_value, html_str

        return max_count_value

class ReasoningResultInterpreter():
    step_name = 'RESULT'
    # ex FINAL_RESULT=RESULT(var=ANSWER0)

    def __init__(self):
        print(f'Registering {self.step_name} step')

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        res_var = parse_result['args']['var']
        output_var = parse_result['output_var']
        assert(step_name==self.step_name)
        return res_var,output_var

    def html(self,output_var,output_txt):
        step_name = html_step_name(self.step_name)
        output_var = html_var_name(output_var)
        output = html_output(output_txt)

        return f"""<div>{output_var}->{step_name}->{output}</div>"""

    def execute(self,prog_step,inspect=False):
        res_var,output_var = self.parse(prog_step)
        output_txt = prog_step.state[res_var]
        if inspect:
            html_str = self.html(output_var,output_txt)
            return output_txt, html_str
        return output_txt

def register_step_interpreters(dataset='log_grounding'):
    if dataset=='log_grounding':
        return dict(
            GetLandmarkSeg=GetLandmarkSegInterpreter(),
            SegAround=SegAroundInterpreter(),
            GetSimilarAreas=GetSimilarAreasInterpreter(),
            GetCluster=GetClusterInterpreter(),
            SegDirection=SegDirectionInterpreter(),
            SegBetween=SegBetweenInterpreter(),
            RESULT=GroundingResultInterpreter()
        )
    elif dataset=='log_estimation':
        return dict(
            GetLandmarkSeg=GetLandmarkSegInterpreter(),
            SegAround=SegAroundInterpreter(),
            GetSimilarAreas=GetSimilarAreasInterpreter(),
            GetCluster=GetClusterInterpreter(),
            SegDirection=SegDirectionInterpreter(),
            SegBetween=SegBetweenInterpreter(),
            MeasureDist=MeasureDistInterpreter(),
            RESULT=EstimationResultInterpreter()
        )
    elif dataset=='log_estimation_height':
        return dict(
            GetLandmarkSeg=GetLandmarkSegInterpreter(),
            SegAround=SegAroundInterpreter(),
            GetSimilarAreas=GetSimilarAreasInterpreter(),
            GetCluster=GetClusterInterpreter(),
            SegDirection=SegDirectionInterpreter(),
            SegBetween=SegBetweenInterpreter(),
            GetAreaHeight=GetAreaHeightInterpreter(),
            RESULT=EstimationResultInterpreter()
        )
    elif dataset=='log_counting':
        return dict(
            GetLandmarkSeg=GetLandmarkSegInterpreter(),
            SegAround=SegAroundInterpreter(),
            GetSimilarAreas=GetSimilarAreasInterpreter(),
            GetCluster=GetClusterInterpreter(),
            SegDirection=SegDirectionInterpreter(),
            SegBetween=SegBetweenInterpreter(),
            ObjCounting=ObjCountingInterpreter(),
            RESULT=CountingResultInterpreter()
        )
    elif dataset=='log_qa':
        return dict(
            GetLandmarkSeg=GetLandmarkSegInterpreter(),
            SegAround=SegAroundInterpreter(),
            GetSimilarAreas=GetSimilarAreasInterpreter(),
            GetCluster=GetClusterInterpreter(),
            SegDirection=SegDirectionInterpreter(),
            SegBetween=SegBetweenInterpreter(),
            ObjCounting=ObjCountingInterpreter(),
            MeasureDist=MeasureDistInterpreter(),
            EVAL=EvalInterpreter(),
            RESULT=ReasoningResultInterpreter()
        )
    elif dataset=='log_reasoning':
        return dict(
            GetLandmarkSeg=GetLandmarkSegInterpreter(),
            SegAround=SegAroundInterpreter(),
            GetSimilarAreas=GetSimilarAreasInterpreter(),
            GetCluster=GetClusterInterpreter(),
            SegDirection=SegDirectionInterpreter(),
            SegBetween=SegBetweenInterpreter(),
            ObjCounting=ObjCountingInterpreter(),
            GetAreaHeight=GetAreaHeightInterpreter(),
            MeasureDist=MeasureDistInterpreter(),
            EVAL=EvalInterpreter(),
            RESULT=ReasoningResultInterpreter()
        )

