import argparse
import dill

from .semantics.type_parsing.ocsf import OcsfDataTypeParser
from .tools.classes import Parser

from .tools.api import BACKEND, Caller
from .tools.OCSF import OCSFSchemaClient


#
def main_debug():
    parser = argparse.ArgumentParser(
        description="Generate parsers for OCSF variables."
    )
    parser.add_argument(
        "saved_debug_file", type=str, help="Path to the parser dill file"
    )
    parser.add_argument(
        "--backend_thread_count",
        type=int,
        help="number of threads to use for backend calls",
        default=1,
    )
    parser.add_argument(
        "--backend",
        choices=BACKEND,
        default=BACKEND[0],
        help="Select the backend to use",
    )

    args = parser.parse_args()
    with open(args.saved_debug_file, "rb") as f:
        templates, elements, ocsf_mappings, examples = dill.load(f)

    caller = Caller(
        args.backend_thread_count,
        backend=args.backend,
        distribute_parallel_requests=True,
    )
    ocsf_client = OCSFSchemaClient(caller)

    ocsf_data_type_parser = OcsfDataTypeParser(ocsf_client, caller)
    parsers = []
    for t, e, mappings, ex in zip(templates, elements, ocsf_mappings, examples):
        for m_grp in mappings:
            for m in m_grp:
                parser = ocsf_data_type_parser.get_parser_for_variable_path(
                    m, t, e, ex
                )
                parsers.append(parser)
                with open("test_state_save2.dill", "wb") as f:
                    dill.dump(parsers, f)

    # with open("test_state.dill", "wb") as f:
    #     dill.dump((template, examples), f)

    # ocsf_data_type_parser.get_parser_for_variable_path(
    #     "ssh_activity.time", template.elements[0], template, examples
    # )
    # ocsf_data_type_parser.get_parser_for_variable_path(
    #     "ssh_activity.device.name", template.elements[1], template, examples
    # )
    # ocsf_data_type_parser.get_parser_for_variable_path(
    #     "ssh_activity.src_endpoint.ip", template.elements[7], template, examples
    # )
    # ocsf_data_type_parser.get_parser_for_variable_path(
    #     "ssh_activity.src_endpoint.port",
    #     template.elements[9],
    #     template,
    #     examples,
    # )


def main():
    main_debug()


def _main():
    # TODO: Plumb this when the mapper is ready
    parser = argparse.ArgumentParser(
        description="Generate parsers for OCSF variables."
    )
    # parser.add_argument("input_file", type=str, help="Path to the input file")
    parser.add_argument(
        "parser_file", type=str, help="Path to the parser dill file"
    )
    parser.add_argument(
        "--backend_thread_count",
        type=int,
        help="number of threads to use for backend calls",
        default=1,
    )
    parser.add_argument(
        "--backend",
        choices=BACKEND,
        default=BACKEND[0],
        help="Select the backend to use",
    )

    args = parser.parse_args()
    with open(args.parser_file, "rb") as f:
        parser: Parser = dill.load(f)

    caller = Caller(
        args.backend_thread_count,
        backend=args.backend,
        distribute_parallel_requests=True,
    )
    ocsf_client = OCSFSchemaClient(caller)

    # with open(args.input_file, "r") as f:
    #     inputs = json.load(f)
    #     template = inputs[0]["template"]
    #     example = inputs[0]["example"]
    # print(example)

    ocsf_data_type_parser = OcsfDataTypeParser(ocsf_client, caller)
    template = parser.tree.gen_template(0)
    examples = parser.entries_per_template[0]
    with open("test_state.dill", "wb") as f:
        dill.dump((template, examples), f)

    ocsf_data_type_parser.get_parser_for_variable_path(
        "ssh_activity.time", template.elements[0], template, examples
    )
    ocsf_data_type_parser.get_parser_for_variable_path(
        "ssh_activity.host.device.name",
        template.elements[1],
        template,
        examples,
    )
    ocsf_data_type_parser.get_parser_for_variable_path(
        "ssh_activity.src_endpoint.ip", template.elements[6], template, examples
    )
    ocsf_data_type_parser.get_parser_for_variable_path(
        "ssh_activity.src_endpoint.port",
        template.elements[8],
        template,
        examples,
    )
