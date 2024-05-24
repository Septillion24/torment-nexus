import numpy as np

def get_feature_names() -> list[str]:
    byte_histogram = [f"Byte Histogram {a}" for a in range(256)] #256

    byte_entropy_histogram = [f"Byte Entropy Histogram {a}" for a in range(256)] #256

    strings_1 = [f"string.{a}" for a in ["numstrings", "avlength", "printables"]]
    strings_2 = [f"string.printabledist_{b}" for b in range(96)]
    strings_3 = [f"string.{a}" for a in ["entropy", "paths", "urls", "registry", "MZ"]] # 8 + 96
    strings = np.concatenate((strings_1,strings_2,strings_3))

    general_info = [f"general.{a}" for a in ["size", "vsize", "has_debug", "exports", "imports", "has_relocations", "has_resources", "has_signature", "has_tls", "symbols"]]

    header_coff = ["header.coff.timestamp"]
    header_coff_machine = [f"header.coff.machine_{a}" for a in range(10)]
    header_coff_characteristics = [f"header.coff.characteristic_{a}" for a in range(10)]
    header_coff_subsystem = [f"header.optional.subsystem_{a}" for a in range(10)]
    header_coff_dll_characteristics = [f"header.optional.dll_characteristic_{a}" for a in range(10)]
    header_coff_magic = [f"header.optional.magic_{a}" for a in range(10)]
    header_optional = [f"header.optional.{a}" for a in ["major_image_version", "minor_image_version", "major_linker_version", "minor_linker_version", "major_operating_system_version", "minor_operating_system_version", "major_subsystem_version", "minor_subsystem_version", "sizeof_code", "sizeof_headers", "sizeof_heap_commit"]] #12
    header = np.concatenate((header_coff,header_coff_machine,header_coff_characteristics,header_coff_subsystem,header_coff_dll_characteristics,header_coff_magic,header_optional))

    sections_general = [f"sections.{a}" for a in ["section_count", "num_empty_sections", "num_unnamed_sections", "num_read_execute_sections", "num_write_sections",]] #JUST general
    sections_section_sizes = [f"sections.section_{a}_size" for a in range(50)] # this messes with hashing which i will understand at a later time
    sections_section_entropy = [f"sections.section_{a}_entropy" for a in range(50)]
    sections_section_vsize = [f"sections.section_{a}_vsize" for a in range(50)]
    sections_entry_name = [f"sections.entry_name_{a}" for a in range(50)]
    sections_characteristics = [f"sections.characteristics_{a}" for a in range(50)]
    sections = np.concatenate((sections_general, sections_section_sizes, sections_section_entropy, sections_section_vsize, sections_entry_name, sections_characteristics))

    imports_libraries = [f"imports.libraries.library_{a}" for a in range(256)]
    imports_imports = [f"imports.import_{a}" for a in range(1024)]
    imports = np.concatenate((imports_libraries,imports_imports))

    exports = [f"exports.export_{a}" for a in range(128)]

    name_order = [a.lower() for a in ["EXPORT_TABLE", "IMPORT_TABLE", "RESOURCE_TABLE", "EXCEPTION_TABLE", "CERTIFICATE_TABLE","BASE_RELOCATION_TABLE", "DEBUG", "ARCHITECTURE", "GLOBAL_PTR", "TLS_TABLE", "LOAD_CONFIG_TABLE","BOUND_IMPORT", "IAT", "DELAY_IMPORT_DESCRIPTOR", "CLR_RUNTIME_HEADER"]]
    data_directories_unflat = [[f"directories.{a}_size", f"directories.{a}_vaddress"] for a in name_order]
    data_directories = [item for sublist in data_directories_unflat for item in sublist]



    feature_names = np.concatenate((byte_histogram, byte_entropy_histogram, strings, general_info, header, sections, imports, exports, data_directories))
    return feature_names