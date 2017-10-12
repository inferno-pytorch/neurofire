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
        # Write to dict
        self.set_random_variable('shifts', shifts)
        self.set_random_variable('origin', originward_leeways)
        self.set_random_variable('shift_or_slide', shift_or_slide)
        self.set_random_variable('slide_from', slide_from)

    def volume_function(self, volume):
        # Build random variables
        self.build_random_variables(num_planes=volume.shape[0], input_image_size=volume.shape[1:])
        # Get'em
        origin = self.get_random_variable('origin')
        shift = self.get_random_variable('shift')
        # TODO Continue

