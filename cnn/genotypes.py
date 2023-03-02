from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PRIMITIVES = [
    'none',
    'skip_connect',
    'sep_conv1_3x3',
    'sep_conv1_5x5',
    'sep_conv1_7x7',
    'sep_conv2_3x3',
    'sep_conv2_5x5',
    'sep_conv2_7x7',
    'sep_conv3_3x3',
    'sep_conv3_5x5',
    'sep_conv3_7x7',
    'sep_conv4_3x3',
    'sep_conv4_5x5',
    'sep_conv4_7x7',
]

NASNet = Genotype(
    normal=[
        ('sep_conv_5x5', 1),
        ('sep_conv_3x3', 0),
        ('sep_conv_5x5', 0),
        ('sep_conv_3x3', 0),
        ('avg_pool_3x3', 1),
        ('skip_connect', 0),
        ('avg_pool_3x3', 0),
        ('avg_pool_3x3', 0),
        ('sep_conv_3x3', 1),
        ('skip_connect', 1),
    ],
    normal_concat=[2, 3, 4, 5, 6],
    reduce=[
        ('sep_conv_5x5', 1),
        ('sep_conv_7x7', 0),
        ('max_pool_3x3', 1),
        ('sep_conv_7x7', 0),
        ('avg_pool_3x3', 1),
        ('sep_conv_5x5', 0),
        ('skip_connect', 3),
        ('avg_pool_3x3', 2),
        ('sep_conv_3x3', 2),
        ('max_pool_3x3', 1),
    ],
    reduce_concat=[4, 5, 6],
)

AmoebaNet = Genotype(
    normal=[
        ('avg_pool_3x3', 0),
        ('max_pool_3x3', 1),
        ('sep_conv_3x3', 0),
        ('sep_conv_5x5', 2),
        ('sep_conv_3x3', 0),
        ('avg_pool_3x3', 3),
        ('sep_conv_3x3', 1),
        ('skip_connect', 1),
        ('skip_connect', 0),
        ('avg_pool_3x3', 1),
    ],
    normal_concat=[4, 5, 6],
    reduce=[
        ('avg_pool_3x3', 0),
        ('sep_conv_3x3', 1),
        ('max_pool_3x3', 0),
        ('sep_conv_7x7', 2),
        ('sep_conv_7x7', 0),
        ('avg_pool_3x3', 1),
        ('max_pool_3x3', 0),
        ('max_pool_3x3', 1),
        ('conv_7x1_1x7', 0),
        ('sep_conv_3x3', 5),
    ],
    reduce_concat=[3, 4, 6]
)

DARTS_V1 = Genotype(
    normal=[
        ('sep_conv_3x3', 1),
        ('sep_conv_3x3', 0),
        ('skip_connect', 0),
        ('sep_conv_3x3', 1),
        ('skip_connect', 0),
        ('sep_conv_3x3', 1),
        ('sep_conv_3x3', 0),
        ('skip_connect', 2)
    ],
    normal_concat=[2, 3, 4, 5],
    reduce=[
        ('max_pool_3x3', 0),
        ('max_pool_3x3', 1),
        ('skip_connect', 2),
        ('max_pool_3x3', 0),
        ('max_pool_3x3', 0),
        ('skip_connect', 2),
        ('skip_connect', 2),
        ('avg_pool_3x3', 0)
    ],
    reduce_concat=[2, 3, 4, 5]
)

DARTS_V2 = Genotype(
    normal=[
        ('sep_conv_3x3', 0),
        ('sep_conv_3x3', 1),
        ('sep_conv_3x3', 0),
        ('sep_conv_3x3', 1),
        ('sep_conv_3x3', 1),
        ('skip_connect', 0),
        ('skip_connect', 0),
        ('dil_conv_3x3', 2)
    ],
    normal_concat=[2, 3, 4, 5],
    reduce=[
        ('max_pool_3x3', 0),
        ('max_pool_3x3', 1),
        ('skip_connect', 2),
        ('max_pool_3x3', 1),
        ('max_pool_3x3', 0),
        ('skip_connect', 2),
        ('skip_connect', 2),
        ('max_pool_3x3', 1)
    ],
    reduce_concat=[2, 3, 4, 5]
)

DARTS_ADP_N2 = Genotype(
    normal=[
        ('max_pool_3x3', 1),
        ('max_pool_3x3', 0),
        ('dil_conv_5x5', 2),
        ('max_pool_3x3', 1)
    ],
    normal_concat=range(2, 4),
    reduce=[
        ('sep_conv_5x5', 0),
        ('max_pool_3x3', 1),
        ('max_pool_3x3', 2),
        ('dil_conv_5x5', 0)
    ],
    reduce_concat=range(2, 4)
)

DARTS_ADP_N3 = Genotype(
    normal=[
        ('max_pool_3x3', 0),
        ('max_pool_3x3', 1),
        ('sep_conv_5x5', 2),
        ('max_pool_3x3', 1),
        ('dil_conv_5x5', 3),
        ('max_pool_3x3', 1)
    ],
    normal_concat=range(2, 5),
    reduce=[
        ('max_pool_3x3', 0),
        ('dil_conv_5x5', 1),
        ('max_pool_3x3', 0),
        ('max_pool_3x3', 2),
        ('skip_connect', 1),
        ('max_pool_3x3', 0)
    ],
    reduce_concat=range(2, 5)
)

DARTS_ADP_N4 = Genotype(
    normal=[
        ('max_pool_3x3', 0),
        ('max_pool_3x3', 1),
        ('max_pool_3x3', 0),
        ('skip_connect', 2),
        ('max_pool_3x3', 0),
        ('max_pool_3x3', 2),
        ('dil_conv_3x3', 4),
        ('max_pool_3x3', 0)
    ],
    normal_concat=range(2, 6),
    reduce=[
        ('max_pool_3x3', 0),
        ('max_pool_3x3', 1),
        ('dil_conv_5x5', 2),
        ('max_pool_3x3', 0),
        ('sep_conv_5x5', 2),
        ('max_pool_3x3', 0),
        ('dil_conv_3x3', 2),
        ('max_pool_3x3', 4)
    ],
    reduce_concat=range(2, 6)
)

CSearch_CIFAR10_ADAS_n3_l3 = Genotype(
    normal=[('sep_conv1_3x3', 0), 
            ('skip_connect', 1), 
            ('sep_conv1_3x3', 2), 
            ('sep_conv4_3x3', 0), 
            ('sep_conv4_3x3', 2), 
            ('sep_conv4_3x3', 1)], 
    normal_concat=range(2, 5), 
    reduce=[('sep_conv4_7x7', 1), 
            ('sep_conv2_5x5', 0), 
            ('sep_conv4_7x7', 1), 
            ('sep_conv4_5x5', 0), 
            ('sep_conv4_3x3', 0), 
            ('sep_conv4_5x5', 3)], 
    reduce_concat=range(2, 5))

test = Genotype(
    normal=[('sep_conv1_3x3', 0), 
            ('skip_connect', 1), 
            ('sep_conv1_3x3', 2), 
            ('sep_conv4_3x3', 0), 
            ('sep_conv4_3x3', 2), 
            ('sep_conv4_3x3', 1)], 
    normal_concat=range(2, 5), 
    reduce=[('sep_conv4_7x7', 1), 
            ('sep_conv2_5x5', 0), 
            ('sep_conv4_7x7', 1), 
            ('sep_conv4_5x5', 0), 
            ('sep_conv4_3x3', 0), 
            ('sep_conv4_5x5', 3)], 
    reduce_concat=range(2, 5))

CSearch_CIFAR10_SGD_n3_l3_stem = Genotype(
    normal=[('sep_conv4_3x3', 0), 
            ('skip_connect', 1), 
            ('sep_conv4_3x3', 0), 
            ('sep_conv1_5x5', 1), 
            ('sep_conv4_5x5', 0), 
            ('sep_conv1_3x3', 1)], 
    normal_concat=range(2, 5), 
    reduce=[('skip_connect', 1), 
            ('sep_conv1_5x5', 0), 
            ('skip_connect', 1), 
            ('sep_conv4_5x5', 2), 
            ('sep_conv4_5x5', 1), 
            ('sep_conv4_7x7', 2)], 
    reduce_concat=range(2, 5))

CSearch_CIFAR10_RMSGD_n4_l4_oldstem = Genotype(
    normal=[('sep_conv3_5x5', 1), 
            ('sep_conv1_5x5', 0), 
            ('sep_conv1_3x3', 1), 
            ('sep_conv1_3x3', 0), 
            ('sep_conv3_7x7', 1), 
            ('sep_conv1_5x5', 0), 
            ('sep_conv1_7x7', 1), 
            ('sep_conv1_3x3', 0)], 
    normal_concat=range(2, 6), 
    reduce=[('sep_conv4_5x5', 1), 
            ('sep_conv1_5x5', 0), 
            ('sep_conv4_5x5', 1), 
            ('sep_conv4_5x5', 2), 
            ('sep_conv4_7x7', 1), 
            ('sep_conv4_3x3', 2), 
            ('sep_conv4_3x3', 1), 
            ('sep_conv4_5x5', 2)], 
    reduce_concat=range(2, 6))

CSearch_CIFAR10_RMSGD_n4_l4_newstem = Genotype(
    normal=[('sep_conv1_5x5', 1), 
            ('sep_conv1_5x5', 0), 
            ('sep_conv1_5x5', 1), 
            ('sep_conv1_7x7', 0), 
            ('sep_conv1_3x3', 1), 
            ('sep_conv1_3x3', 0), 
            ('sep_conv1_5x5', 1), 
            ('sep_conv1_3x3', 0)], 
    normal_concat=range(2, 6), 
    reduce=[('sep_conv4_5x5', 1), 
            ('sep_conv4_7x7', 0), 
            ('sep_conv4_7x7', 1), 
            ('sep_conv1_7x7', 0), 
            ('sep_conv4_3x3', 1), 
            ('sep_conv4_5x5', 2), 
            ('sep_conv4_5x5', 1), 
            ('sep_conv4_3x3', 2)], 
    reduce_concat=range(2, 6))

CSearch_CIFAR10_SGD_n4_l4 = Genotype(
    normal=[('sep_conv1_5x5', 1), 
            ('sep_conv1_5x5', 0), 
            ('sep_conv1_3x3', 1), 
            ('sep_conv4_3x3', 0), 
            ('sep_conv1_7x7', 1), 
            ('sep_conv4_3x3', 3), 
            ('sep_conv1_3x3', 1), 
            ('sep_conv1_3x3', 2)], 
    normal_concat=range(2, 6), 
    reduce=[('sep_conv4_7x7', 1), 
            ('sep_conv2_5x5', 0), 
            ('sep_conv4_5x5', 1), 
            ('sep_conv1_3x3', 0), 
            ('sep_conv4_3x3', 1), 
            ('sep_conv4_5x5', 2), 
            ('sep_conv4_5x5', 1), 
            ('sep_conv4_3x3', 2)], 
    reduce_concat=range(2, 6))

CSearch_CIFAR10_RMSGD_n5_l4_newstem = Genotype(
    normal=[('sep_conv1_5x5', 1), 
            ('skip_connect', 0), 
            ('skip_connect', 0), 
            ('sep_conv1_5x5', 1), 
            ('sep_conv4_3x3', 0), 
            ('sep_conv1_7x7', 1), 
            ('sep_conv4_5x5', 0), 
            ('sep_conv1_5x5', 1), 
            ('sep_conv1_5x5', 1), 
            ('sep_conv4_3x3', 4)], 
    normal_concat=range(2, 7), 
    reduce=[('sep_conv4_3x3', 1), 
            ('sep_conv1_3x3', 0), 
            ('sep_conv4_3x3', 1), 
            ('sep_conv1_5x5', 0), 
            ('sep_conv4_3x3', 1), 
            ('sep_conv4_3x3', 2), 
            ('sep_conv4_3x3', 1), 
            ('sep_conv4_3x3', 2), 
            ('sep_conv4_5x5', 1), 
            ('sep_conv4_3x3', 2)], 
    reduce_concat=range(2, 7))
 

# CSearch_CIFAR10_RMSGD_n5_l4_strided = Genotype(
#     normal=[('sep_conv1_strided_5x5', 1), 
#             ('skip_connect', 0), 
#             ('skip_connect', 0), 
#             ('sep_conv1_strided_5x5', 1), 
#             ('sep_conv4_strided_3x3', 0), 
#             ('sep_conv1_7x7', 1), 
#             ('sep_conv4_strided_5x5', 0), 
#             ('sep_conv1_strided_5x5', 1), 
#             ('sep_conv1_strided_5x5', 1), 
#             ('sep_conv4_strided_3x3', 4)], 
#     normal_concat=range(2, 7), 
#     reduce=[('sep_conv4_strided_3x3', 1), 
#             ('sep_conv1_strided_3x3', 0), 
#             ('sep_conv4_strided_3x3', 1), 
#             ('sep_conv1_strided_5x5', 0), 
#             ('sep_conv4_strided_3x3', 1), 
#             ('sep_conv4_strided_3x3', 2), 
#             ('sep_conv4_strided_3x3', 1), 
#             ('sep_conv4_strided_3x3', 2), 
#             ('sep_conv4_strided_5x5', 1), 
#             ('sep_conv4_strided_3x3', 2)], 
#     reduce_concat=range(2, 7))
 
NEWCONV_design_cin4_cifar10_DARTSsettings = Genotype(
     normal=[('sep_conv1_7x7', 0), 
     ('sep_conv1_3x3', 1), 
     ('sep_conv4_3x3', 2), 
     ('sep_conv1_3x3', 0), 
     ('sep_conv4_3x3', 3), 
     ('sep_conv1_7x7', 0), 
     ('sep_conv4_3x3', 2), 
     ('sep_conv1_7x7', 0)], 
     normal_concat=range(2, 6), 
     reduce=[('sep_conv4_3x3', 1), 
     ('sep_conv4_7x7', 0), 
     ('sep_conv4_7x7', 0), 
     ('sep_conv4_3x3', 2), 
     ('sep_conv4_7x7', 0), 
     ('sep_conv4_5x5', 2), 
     ('sep_conv4_5x5', 1), 
     ('sep_conv4_7x7', 0)], 
     reduce_concat=range(2, 6))

NEWCONV_MAXPOOL = Genotype(
    normal=[('max_pool_3x3', 1), 
            ('max_pool_3x3', 0), 
            ('max_pool_3x3', 1), 
            ('max_pool_3x3', 0), 
            ('max_pool_3x3', 1), 
            ('max_pool_3x3', 0), 
            ('max_pool_3x3', 1), 
            ('max_pool_3x3', 0)], 
    normal_concat=range(2, 6), 
    reduce=[('max_pool_3x3', 0), 
            ('dil_conv_5x5', 1), 
            ('max_pool_3x3', 0), 
            ('max_pool_3x3', 2), 
            ('dil_conv_3x3', 3), 
            ('sep_conv_5x5', 2), 
            ('sep_conv_3x3', 1), 
            ('dil_conv_3x3', 4)], 
    reduce_concat=range(2, 6))

NEWCONV_RMSGD_Searched = Genotype(
        normal=[('dil_conv_5x5', 1), 
                    ('max_pool_3x3', 0), 
                    ('max_pool_3x3', 0), 
                    ('dil_conv_5x5', 1), 
                    ('dil_conv_5x5', 3), 
                    ('dil_conv_5x5', 2), 
                    ('sep_conv_3x3', 3), 
                    ('dil_conv_5x5', 2)], 
            normal_concat=range(2, 6), 
            reduce=[('dil_conv_5x5', 0), 
                    ('dil_conv_5x5', 1), 
                    ('dil_conv_5x5', 1), 
                    ('dil_conv_5x5', 2), 
                    ('dil_conv_3x3', 3), 
                    ('dil_conv_5x5', 1), 
                    ('sep_conv_5x5', 3),                                                                                                                                                                                                    ('dil_conv_3x3', 4)], 
            reduce_concat=range(2, 6))


NEWCONV_RMSGD_100EPOCH = Genotype(
    normal=[('max_pool_3x3', 0), 
            ('dil_conv_5x5', 1), 
            ('max_pool_3x3', 0), 
            ('max_pool_3x3', 2), 
            ('dil_conv_3x3', 1), 
            ('max_pool_3x3', 0), 
            ('dil_conv_5x5', 3), 
            ('dil_conv_5x5', 4)], 
    normal_concat=range(2, 6), 
            reduce=[('skip_connect', 0), 
                    ('max_pool_3x3', 1), 
                    ('skip_connect', 0), 
                    ('sep_conv_5x5', 1), 
                    ('dil_conv_3x3', 0), 
                    ('max_pool_3x3', 3), 
                    ('sep_conv_3x3', 4),                                                                                                                                                                                                ('max_pool_3x3', 2)], 
         reduce_concat=range(2, 6))

NEWCONV_RMSGD_N5_50EPOCH = Genotype(
    normal=[('dil_conv_3x3', 1), 
            ('dil_conv_3x3', 0), 
            ('dil_conv_5x5', 1), 
            ('dil_conv_3x3', 0), 
            ('sep_conv_5x5', 2), 
            ('dil_conv_5x5', 1), 
            ('dil_conv_5x5', 4), 
            ('dil_conv_5x5', 1), 
            ('dil_conv_5x5', 5), 
            ('sep_conv_3x3', 3)], 
    normal_concat=range(2, 7), 
    reduce=[('dil_conv_3x3', 1), 
            ('dil_conv_5x5', 0), 
            ('dil_conv_3x3', 0), 
            ('max_pool_3x3', 1), 
            ('dil_conv_3x3', 0), 
            ('sep_conv_3x3', 3), 
            ('dil_conv_5x5', 4), 
            ('dil_conv_5x5', 3), 
            ('dil_conv_3x3', 0), 
            ('sep_conv_3x3', 3)], 
    reduce_concat=range(2, 7))