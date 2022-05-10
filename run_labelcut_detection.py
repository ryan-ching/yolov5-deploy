import os
import PIL.Image
import PIL.ImageOps
from collections import namedtuple


def image_crop_row_v2(im_obj, filename, min_leftover=10):
    filename_dum, file_extension_dum = os.path.splitext(filename)

    width, height = im_obj.size
    n_sub_img = height // width

    filename_sq_lst = []
    # filename_nsq_lst = []
    im_sq_lst = []

    left = 0
    right = width
    for i in range(0, n_sub_img):
        # Setting the points for cropped image
        top = i * width
        bottom = (i + 1) * width

        # Cropping image
        im_cropped = im_obj.crop((left, top, right, bottom))
        filename_sq = filename_dum + '_croprow_sq' + str(i) + file_extension_dum
        filename_sq_lst.append(filename_sq)
        im_sq_lst.append(im_cropped)
        # im_cropped.save(output_folder + filename_sq)

    filename_nsq = ''
    im_nsq = ''
    if height % width > min_leftover:  # ????
        i = i + 1
        top = top + width
        bottom = height
        im_nsq = im_obj.crop((left, top, right, bottom))
        filename_nsq = filename_dum + '_croprow_nsq' + str(i) + file_extension_dum
        # filename_nsq_lst.append(filename_nsq)
        # im_cropped.save(output_folder + filename_nsq)

    return {'square image name': filename_sq_lst, 'square image object': im_sq_lst,
            'non-square image name': filename_nsq, 'non-square image object': im_nsq}


def image_crop_column_v2(im_obj, filename, min_leftover=10):
    filename_dum, file_extension_dum = os.path.splitext(filename)

    width, height = im_obj.size
    n_sub_img = width // height

    filename_sq_lst = []
    # filename_nsq_lst = []
    im_sq_lst = []

    top = 0
    bottom = height
    for i in range(0, n_sub_img):
        # Setting the points for cropped image
        left = i * height
        right = (i + 1) * height

        # Cropping image
        im_cropped = im_obj.crop((left, top, right, bottom))
        filename_sq = filename_dum + '_cropcol_sq' + str(i) + file_extension_dum
        filename_sq_lst.append(filename_sq)
        im_sq_lst.append(im_cropped)
        # im_cropped.save(output_folder + filename_sq)

    filename_nsq = ''
    im_nsq = ''
    if width % height > min_leftover:
        i = i + 1
        left = left + height
        right = width
        im_nsq = im_obj.crop((left, top, right, bottom))
        filename_nsq = filename_dum + '_cropcol_nsq' + str(i) + file_extension_dum
        # filename_nsq_lst.append(filename_nsq)
        # im_cropped.save(output_folder + filename_nsq)

    return {'square image name': filename_sq_lst, 'square image object': im_sq_lst,
            'non-square image name': filename_nsq, 'non-square image object': im_nsq}


def padding(im_obj, desired_size):
    width, height = im_obj.size
    if desired_size < width or desired_size < height:
        raise ValueError('The desired_size should NOT be less than the width or height of the input image.')
    delta_width = desired_size - width
    delta_height = desired_size - height
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    pad = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
    return PIL.ImageOps.expand(im_obj, pad)


def image_preprocessing_v3(filename_main, input_folder_main, output_folder_main='', model_img_size=640,
                           min_acceptable_image_size=50, min_aspect_ratio_to_crop=1.6, min_leftover_image=10,
                           save_subimages=False):
    im = PIL.Image.open(os.path.join(input_folder_main, filename_main)).convert('RGB')
    im = PIL.ImageOps.exif_transpose(im)
    # im.show()
    width, height = im.size

    if save_subimages and output_folder_main == '':
        raise ValueError('With save_subimages=True you must specify a valid output_folder_main.')

    imgobj_all_lst = []
    if width > min_acceptable_image_size and height > min_acceptable_image_size:

        if width > height:
            if width > (min_aspect_ratio_to_crop * height):
                op_crop = image_crop_column_v2(im_obj=im, filename=filename_main, min_leftover=min_leftover_image)
                for i, im_dum in enumerate(op_crop['square image object']):
                    # im_crop_sq = PIL.Image.open(temp_folder_main + dum).convert('RGB')
                    im_crop_sq_resize = im_dum.resize((model_img_size, model_img_size), PIL.Image.ANTIALIAS)
                    if save_subimages:
                        # saving image
                        dum_filename, dum_extension = os.path.splitext(op_crop['square image name'][i])
                        im_crop_sq_resize.save(output_folder_main + dum_filename + '_resize_final' + dum_extension)
                    else:
                        # appending image
                        imgobj_all_lst.append(im_crop_sq_resize)

                dum_nonsq_filename = op_crop['non-square image name']
                if dum_nonsq_filename != '':
                    im_crop_nsq = op_crop['non-square image object']
                    # im_crop_nsq = PIL.Image.open(temp_folder_main + dum).convert('RGB')
                    im_crop_nsq_pad = padding(im_obj=im_crop_nsq, desired_size=max(im_crop_nsq.size))
                    im_crop_nsq_pad_resize = im_crop_nsq_pad.resize((model_img_size, model_img_size),
                                                                    PIL.Image.ANTIALIAS)
                    if save_subimages:
                        # saving image
                        dum_filename, dum_extension = os.path.splitext(dum_nonsq_filename)
                        im_crop_nsq_pad_resize.save(
                            output_folder_main + dum_filename + '_pad_resize_final' + dum_extension)
                    else:
                        # appending image
                        imgobj_all_lst.append(im_crop_nsq_pad_resize)

            else:
                im_pad = padding(im_obj=im, desired_size=max(im.size))
                im_pad_resize = im_pad.resize((model_img_size, model_img_size), PIL.Image.ANTIALIAS)
                if save_subimages:
                    # saving image
                    dum_filename, dum_extension = os.path.splitext(filename_main)
                    im_pad_resize.save(output_folder_main + dum_filename + '_pad_resize_final' + dum_extension)
                else:
                    # appending image
                    imgobj_all_lst.append(im_pad_resize)

        else:
            if height > (min_aspect_ratio_to_crop * width):
                op_crop = image_crop_row_v2(im_obj=im, filename=filename_main, min_leftover=min_leftover_image)
                for i, im_dum in enumerate(op_crop['square image object']):
                    # im_crop_sq        = PIL.Image.open(temp_folder_main + dum).convert('RGB')
                    im_crop_sq_resize = im_dum.resize((model_img_size, model_img_size), PIL.Image.ANTIALIAS)
                    if save_subimages:
                        # saving image
                        dum_filename, dum_extension = os.path.splitext(op_crop['square image name'][i])
                        im_crop_sq_resize.save(output_folder_main + dum_filename + '_resize_final' + dum_extension)
                    else:
                        # appending image
                        imgobj_all_lst.append(im_crop_sq_resize)

                dum_nonsq_filename = op_crop['non-square image name']
                if dum_nonsq_filename != '':
                    im_crop_nsq = op_crop['non-square image object']
                    # im_crop_nsq = PIL.Image.open(temp_folder_main + dum).convert('RGB')
                    im_crop_nsq_pad = padding(im_obj=im_crop_nsq, desired_size=max(im_crop_nsq.size))
                    im_crop_nsq_pad_resize = im_crop_nsq_pad.resize((model_img_size, model_img_size),
                                                                    PIL.Image.ANTIALIAS)
                    if save_subimages:
                        # saving image
                        dum_filename, dum_extension = os.path.splitext(dum_nonsq_filename)
                        im_crop_nsq_pad_resize.save(
                            output_folder_main + dum_filename + '_pad_resize_final' + dum_extension)
                    else:
                        # appending image
                        imgobj_all_lst.append(im_crop_nsq_pad_resize)

            else:
                im_pad = padding(im_obj=im, desired_size=max(im.size))
                im_pad_resize = im_pad.resize((model_img_size, model_img_size), PIL.Image.ANTIALIAS)
                if save_subimages:
                    # saving image
                    dum_filename, dum_extension = os.path.splitext(filename_main)
                    im_pad_resize.save(output_folder_main + dum_filename + '_pad_resize_final' + dum_extension)
                else:
                    # appending image
                    imgobj_all_lst.append(im_pad_resize)

    return imgobj_all_lst


# pred_type='best' will only return the highest confidence prediction in case multiple occurrences of labelcut are detected
# pred_type='all' will return all prediction in case multiple occurrences of labelcut are detected
def run_labelcut_detection(model, image_filename, image_in_folder, model_train_image_size,
                           image_out_folder, pred_type='max', save_images=False):

    subimg_lst = image_preprocessing_v3(filename_main=image_filename,
                                        input_folder_main=image_in_folder,
                                        output_folder_main='',
                                        model_img_size=model_train_image_size,
                                        min_acceptable_image_size=50,
                                        min_aspect_ratio_to_crop=1.6,
                                        min_leftover_image=10,
                                        save_subimages=False)

    results_all = model(subimg_lst, size=model_train_image_size)

    labelcut_list = []
    xdims = namedtuple("xdims", "xmin xmax")
    ydims = namedtuple("ydims", "ymin ymax")

    message = 'No Labelcut detected!!'
    for res in results_all.pandas().xyxy:
        if res.shape[0] > 0:
            message = 'Labelcut detected!'
            for row in res.itertuples():
                labelcut_list.append({'confidence': row.confidence,
                                      'coordinates': [xdims(xmin=row.xmin, xmax=row.xmax), ydims(ymin=row.ymin, ymax=row.ymax)]})

    # sorting the predicted occurrences of labelcut by confidence score in descending order
    labelcut_list.sort(key=lambda x: x['confidence'], reverse=True)

    if message == 'Labelcut detected!':
        if save_images:
            filename_noext, _ = os.path.splitext(image_filename)
            results_all.save(os.path.join(image_out_folder, filename_noext))

        if pred_type == 'max':
            return {"detected": True, "predictions": labelcut_list[0]}
        elif pred_type == 'all':
            return {"detected": True, "predictions": labelcut_list}
        else:
            print("invalid pred_type='{}' set. expected values = ['max', 'all']". format(pred_type))
            return None

    else:
        return {"detected": False, "predictions": None}