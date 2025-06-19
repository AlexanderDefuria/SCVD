from pathlib import Path
import os

ROOT_DIR = Path(__file__).parent.parent
JOERN_OUTPUT_CACHE = ROOT_DIR / "data" /  ".cache" / "joern_output"
JOERN_OUTPUT_TEMP = ROOT_DIR / "data" /  ".cache" / "joern_output_temp"
JOERN_INTERMIDATE_GRAPH = ROOT_DIR / "data" / ".cache" / "joern_intermediate_graph" # CFG with assignments that have already been parsed and are ready to build a abstraction dictionary but still contain the original code statemtents in "ABSTRACTION".
VERBOSE = " > /dev/null 2>&1"
# VERBOSE = ""  # Set to empty string to enable verbose output

os.makedirs(JOERN_OUTPUT_CACHE, exist_ok=True)
os.makedirs(JOERN_OUTPUT_TEMP, exist_ok=True)
os.makedirs(JOERN_INTERMIDATE_GRAPH, exist_ok=True)


EXAMPLE_SC = """
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


