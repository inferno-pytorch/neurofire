import numpy as np
import torch

from inferno.io.transform import Transform


class RandomSlide(Transform):
    """Transform to randomly sample misalignments."""
    def __init__(self, output_image_size, shift_vs_slide_proba=0.5, **super_kwargs):
        super(RandomSlide, self).__init__(**super_kwargs)
        # Make sure we have a 2-tuple
        output_image_size = (output_image_size, output_image_size) \
            if isinstance(output_image_size, int) else output_image_size
        assert len(output_image_size) == 2, \
            "Inconsistent output_image_size: {}. Must have 2 elements.".format(output_image_size)
        self.output_image_size = tuple(output_image_size)
        self.shift_vs_slide_proba = shift_vs_slide_proba

    def build_random_variables(self, num_planes, input_image_size):
        # Compute the available slide leeways. We have two sets of leeways - origin-ward
        # (i.e. close to the origin at top left of an image) and antiorigin-ward
        # (closer to the bottom right)
        originward_leeways = tuple((_input_size - _output_size) // 2
                                   for _input_size, _output_size in
                                   zip(input_image_size, self.output_image_size))
        antioriginward_leeways = tuple(_input_size - _output_size - _originward
                                       for _input_size, _output_size, _originward in
                                       zip(input_image_size,
                                           self.output_image_size,
                                           originward_leeways))
        # We have: leeways[0] = (originward, antioriginward)
        leeways = tuple(zip(originward_leeways, antioriginward_leeways))
        # Now to sample the shifts, we fix our origin on the top left corner of the input image.
        # From there on, we could go a max orginward_leeway to the left or antioriginward_leeway to
        # the right.
        shifts = tuple(np.random.randint(low=-leeway[0], high=leeway[1]) for leeway in leeways)
        # Select whether to shift or slide
        shift_or_slide = 'shift' if np.random.uniform() < self.shift_vs_slide_proba else 'slide'
        # Select from which plane on to slide
        slide_from = np.random.randint(low=1, high=num_planes)
        shift_at = np.random.randint(low=0, high=num_planes)
        # Write to dict
        self.set_random_variable('shifts', shifts)
        self.set_random_variable('origin', originward_leeways)
        self.set_random_variable('shift_or_slide', shift_or_slide)
        self.set_random_variable('slide_from', slide_from)
        self.set_random_variable('shift_at', shift_at)

    def shift_and_crop(self, image, zero_shift=False):
        # Get random variables
        origin = self.get_random_variable('origin')
        shifts = self.get_random_variable('shifts')
        # Kill shift if requested
        if zero_shift:
            shifts = (0, 0)
        # Get slice
        starts = tuple(_origin + _shift for _origin, _shift in zip(origin, shifts))
        stops = tuple(_start + _size for _start, _size in zip(starts, self.output_image_size))
        slices = tuple(slice(_start, _stop) for _start, _stop in zip(starts, stops))
        # Crop and return
        return image[slices]

    def volume_function(self, volume):
        # TODO Validate volume shape
        # Build random variables
        self.build_random_variables(num_planes=volume.shape[0],
                                    input_image_size=volume.shape[1:])
        # Get random variables
        shift_or_slide = self.get_random_variable('shift_or_slide')
        # Shift or slide?
        if shift_or_slide == 'shift':
            # Shift
            shift_at = self.get_random_variable('shift_at')
            # Don't shift if plane_num doesn't equal shift_at
            out_volume = np.array([self.shift_and_crop(image=plane,
                                                       zero_shift=(plane_num != shift_at))
                                   for plane_num, plane in enumerate(volume)])
        else:
            # Slide
            slide_from = self.get_random_variable('slide_from')
            # Don't shift if plane_num isn't larger than or equal to slide_from
            out_volume = np.array([self.shift_and_crop(image=plane,
                                                       zero_shift=(plane_num < slide_from))
                                   for plane_num, plane in enumerate(volume)])
        # Done
        return out_volume

