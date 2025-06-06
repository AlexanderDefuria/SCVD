EXAMPLE_A = """
#define TEST 1

int FF_ARRAY_ELEMS(int *arr) {
    return sizeof(arr) / sizeof(arr[0]);
}

float x = 0.0f;

int s337m_get_offset_and_codec(void *avctx, uint64_t state, int data_type, int data_size, int *offset, void *codec);

char *str = "Hello, World!";
char *str_two_elctric_boogaloo = NULL;
int check = TEST;

static int s337m_probe(AVProbeData *p)
{
    uint64_t state = 0;
    int markers[3] = { 0 };
    int i, pos, sum, max, data_type, data_size, offset;
    uint8_t *buf;
    int *test = (int*) malloc(4);
    int a_test = 1;
    float b_test = (float) a_test;

    for (pos = 0; pos < p->buf_size; pos++) {
        state = (state << 8) | p->buf[pos];
        if (!IS_LE_MARKER(state))
            continue;

        buf = p->buf + pos + 1;
        if (IS_16LE_MARKER(state)) {
            data_type = AV_RL16(buf    );
            data_size = AV_RL16(buf + 2);
        } else {
            data_type = AV_RL24(buf    );
            data_size = AV_RL24(buf + 3);
        }

        if (s337m_get_offset_and_codec(NULL, state, data_type, data_size, &offset, NULL))
            continue;

        i = IS_16LE_MARKER(state) ? 0 : IS_20LE_MARKER(state) ? 1 : 2;
        markers[i]++;

        pos  += IS_16LE_MARKER(state) ? 4 : 6;
        pos  += offset;
        state = 0;
    }

    sum = max = 0;
    for (i = 0; i < FF_ARRAY_ELEMS(markers); i++) {
        sum += markers[i];
        if (markers[max] < markers[i])
            max = i;
    }

    if (markers[max] > 3 && markers[max] * 4 > sum * 3)
        return AVPROBE_SCORE_EXTENSION + 1;

    return 0;
}
"""

RESULT_A = {
    '30064771076': {'constant': 2, 'api_call': 0, 'data_type': 2, 'operator': 0},
    '30064771077': {'constant': 4, 'api_call': 0, 'data_type': 4, 'operator': 9},
    '30064771079': {'constant': 2, 'api_call': 0, 'data_type': 4, 'operator': 6},
    '30064771081': {'constant': 5, 'api_call': 5, 'data_type': 6, 'operator': 5},
    '30064771084': {'constant': 3, 'api_call': 0, 'data_type': 0, 'operator': 0},
    '30064771085': {'constant': 0, 'api_call': 0, 'data_type': 5, 'operator': 5},
    '30064771087': {'constant': 2, 'api_call': 0, 'data_type': 0, 'operator': 0},
    '30064771090': {'constant': 0, 'api_call': 0, 'data_type': 0, 'operator': 2},
    '30064771091': {'constant': 9, 'api_call': 0, 'data_type': 2, 'operator': 10},
    '30064771098': {'constant': 3, 'api_call': 0, 'data_type': 8, 'operator': 3},
    '30064771103': {'constant': 0, 'api_call': 4, 'data_type': 0, 'operator': 0},
    '30064771105': {'constant': 7, 'api_call': 4, 'data_type': 0, 'operator': 3},
    '30064771108': {'constant': 0, 'api_call': 3, 'data_type': 0, 'operator': 0},
    '30064771110': {'constant': 4, 'api_call': 3, 'data_type': 0, 'operator': 3},
    '30064771115': {'constant': 2, 'api_call': 2, 'data_type': 0, 'operator': 4},
    '30064771120': {'constant': 0, 'api_call': 0, 'data_type': 7, 'operator': 2},
    '30064771122': {'constant': 5, 'api_call': 2, 'data_type': 0, 'operator': 4},
    '30064771125': {'constant': 0, 'api_call': 0, 'data_type': 0, 'operator': 0},
    '30064771126': {'constant': 2, 'api_call': 0, 'data_type': 2, 'operator': 0},
    '30064771127': {'constant': 2, 'api_call': 0, 'data_type': 0, 'operator': 8},
    '30064771128': {'constant': 2, 'api_call': 0, 'data_type': 0, 'operator': 0},
    '30064771129': {'constant': 2, 'api_call': 0, 'data_type': 0, 'operator': 0},
    '30064771132': {'constant': 0, 'api_call': 0, 'data_type': 0, 'operator': 2},
    '30064771133': {'constant': 0, 'api_call': 0, 'data_type': 0, 'operator': 7},
    '30064771138': {'constant': 0, 'api_call': 0, 'data_type': 0, 'operator': 0},
    '30064771147': {'constant': 8, 'api_call': 0, 'data_type': 5, 'operator': 0},
    '30064771148': {'constant': 6, 'api_call': 0, 'data_type': 3, 'operator': 0},
    '30064771149': {'constant': 0, 'api_call': 0, 'data_type': 3, 'operator': 0},
    '30064771150': {'constant': 3, 'api_call': 6, 'data_type': 0, 'operator': 0}
}

