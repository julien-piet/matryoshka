import bs4
import requests


def parse_udm_html(html_content):
    """Parse UDM HTML documentation and extract field information into a dictionary structure.

    Args:
        html_content: HTML content as string

    Returns:
        Dictionary with UDM field paths as keys and field info as values
    """
    soup = bs4.BeautifulSoup(html_content, "html.parser")
    result = {}

    # Load all enums
    enums = get_enum_types(soup)

    # Load basic types
    basic_types = get_basic_types(soup)

    # Find and parse all tables with field definitions
    all_tables = {}
    # Find all h3 and h2 headings that might have tables
    for heading in soup.find_all(["h2", "h3"]):
        heading_id = heading.get("id", "")
        heading_text = heading.get_text(strip=True)

        # Look for the next table after this heading
        next_element = heading.find_next_sibling()
        while next_element:
            if next_element.name in ["h2", "h3"]:
                break
            if next_element.name == "div":
                table = next_element.find("table")
                if table and table.find("thead"):
                    # Check if this is a field definition table
                    headers = [
                        th.get_text(strip=True)
                        for th in table.find("thead").find_all("th")
                    ]
                    if "Field Name" in headers and "Type" in headers:
                        all_tables[heading_text] = {
                            "id": heading_id,
                            "table": table,
                            "heading_level": heading.name,
                        }
                        break
            next_element = next_element.find_next_sibling()

    # Convert all tables to a dictionary
    parsed_tables = {}
    for heading, info in all_tables.items():
        parsed_tables[heading] = parse_fields_from_table(
            info["table"], enums, basic_types
        )

    # Parse the main UDM event table
    recursive_parse("event", "UDM Event data model", result, parsed_tables)

    return result, enums, basic_types


def recursive_parse(
    field_name, table_name, result, parsed_tables, prefix="", type_list=None
):
    """Recursively parse a table and add fields to the result dictionary."""
    # Check if the table exists in parsed tables
    if table_name not in parsed_tables:
        raise ValueError(f"Table '{table_name}' not found in parsed tables.")

    # Initialize type_list if not provided
    if type_list is None:
        type_list = []
    else:
        type_list = type_list[:]

    # Avoid infinite recursion
    if table_name in type_list:
        return

    # Update the prefix and type list
    if not prefix:
        prefix = field_name.lower()
    else:
        prefix = f"{prefix}.{field_name.lower()}"
    type_list.append(table_name)

    # Add the current table to the result dictionary
    table = parsed_tables[table_name]
    for field, info in table.items():
        # Create the full path for the field
        full_path = f"{prefix}.{field}"

        # Add the field to the result
        result[full_path] = {
            "Description": info.get("Description", ""),
            "Type": info.get("Type", ""),
            "Label": info.get("Label", ""),
            "Enum": info.get("Enum", False),
            "Object": info.get("Object", False),
            "Link": info.get("Link", None),
        }

        # If the type is an object, recursively parse its fields
        if info.get("Type", None) in parsed_tables:
            if not info.get("Object", False):
                raise ValueError(
                    f"Expected object type for {full_path}, but got"
                    f" {info.get('Type', None)}"
                )
            recursive_parse(
                field,
                info.get("Type", None),
                result,
                parsed_tables,
                prefix,
                type_list,
            )


def parse_fields_from_table(table, enums, basic_types):
    """Parse fields from a table and add them to result with the given prefix."""
    rows = table.find("tbody").find_all("tr")
    result = {}
    for row in rows:
        cells = row.find_all("td")
        if len(cells) >= 4:
            field_name = cells[0].get_text(strip=True)
            field_type = extract_type_info(cells[1], enums)
            label = cells[2].get_text(strip=True)
            description = clean_description(cells[3].get_text(strip=True))
            add_field_to_result(
                result, field_name, field_type, label, description, basic_types
            )
    return result


def add_field_to_result(
    result, path, field_type, label, description, basic_types
):
    """Add a field to the result dictionary."""
    # Determine if it's an object type
    type_name = field_type["type_name"]
    is_enum = field_type["is_enum"]
    type_value = type_name

    # Check if it's a basic type or object
    if is_enum:
        is_object = False
    elif type_name in basic_types:
        is_object = False
    elif field_type.get("link") and "https" in field_type.get("link", ""):
        # External type
        is_object = False
    else:
        # Complex type - mark as object
        is_object = True

    field_info = {
        "Description": description,
        "Type": type_value,
        "Label": label,
        "Enum": is_enum,
        "Object": is_object,
    }

    if field_type.get("link"):
        field_info["Link"] = field_type["link"]

    result[path] = field_info


def extract_type_info(type_cell, enums):
    """Extract type information from a type cell."""
    # Get type name
    if type_cell.find("a"):
        text = type_cell.find("a").get_text(strip=True)
    else:
        text = type_cell.get_text(strip=True)

    # Check if it's an enumeration
    is_enum = text in enums

    # Check for external links
    link = None
    a_tag = type_cell.find("a")
    if a_tag and a_tag.get("href") and "https://" in a_tag["href"]:
        link = a_tag["href"]

    return {"type_name": text, "is_enum": is_enum, "link": link}


def clean_description(description):
    """Clean up description text.
    Remove extra whitespace and newlines
    """
    description = " ".join(description.split())
    return description


def get_basic_types(soup=None):
    """Return a set of basic/primitive types."""
    tables = soup.find_all("table")
    for table in tables:
        headers = [
            th.get_text(strip=True).lower()
            for th in table.find("thead").find_all("th")
        ]
        if "datatype" in headers:
            result = set()
            rows = table.find("tbody").find_all("tr")
            for row in rows:
                cells = row.find_all("td")
                if cells:
                    result.add(cells[0].get_text(strip=True))
            return result

    return {
        "string",
        "bool",
        "int32",
        "int64",
        "uint32",
        "uint64",
        "float",
        "double",
        "bytes",
        "fixed32",
        "fixed64",
        "sfixed32",
        "sfixed64",
        "sint32",
        "sint64",
    }


def get_enum_types(soup):
    # First, let's find and parse all tables with enum definitions
    all_tables = {}
    # Find all h3 and h2 headings that might have tables
    for heading in soup.find_all(["h2", "h3"]):
        heading_id = heading.get("id", "")
        heading_text = heading.get_text(strip=True)

        # Look for the next table after this heading
        next_element = heading.find_next_sibling()
        while next_element:
            if next_element.name in ["h2", "h3"]:
                break
            if next_element.name == "div":
                table = next_element.find("table")
                if table and table.find("thead"):
                    # Check if this is a field definition table
                    headers = [
                        th.get_text(strip=True)
                        for th in table.find("thead").find_all("th")
                    ]
                    if "Enum Value" in headers:
                        all_tables[heading_text] = {
                            "id": heading_id,
                            "table": table,
                            "heading_level": heading.name,
                        }
                        break
            next_element = next_element.find_next_sibling()

    enums = {}
    for heading, info in all_tables.items():
        enums[heading] = parse_enum_table(info["table"])

    return enums


def parse_enum_table(table):
    """Parse an enum table and add enum definitions to the result."""
    rows = table.find("tbody").find_all("tr")
    result = {}
    for row in rows:
        cells = row.find_all("td")
        if len(cells) >= 2:
            enum_name = cells[0].get_text(strip=True)
            enum_value = cells[1].get_text(strip=True)
            description = (
                clean_description(cells[2].get_text(strip=True))
                if len(cells) > 2
                else ""
            )
            # Add to result
            result[enum_name] = {
                "Value": enum_value,
                "Description": description,
            }
    return result


def extract(filename=None):
    """Fetch and parse the UDM HTML. If filename is provided, read from disk; otherwise fetch from the official URL."""
    if filename:
        with open(filename, "r", encoding="utf-8") as file:
            html_content = file.read()
    else:
        # Download the HTML content
        url = "https://cloud.google.com/chronicle/docs/reference/udm-field-list"
        response = requests.get(url)
        response.raise_for_status()  # Raise an error if the request failed
        html_content = response.text

    return parse_udm_html(html_content)


__all__ = [
    "parse_udm_html",
    "recursive_parse",
    "parse_fields_from_table",
    "add_field_to_result",
    "extract_type_info",
    "clean_description",
    "get_basic_types",
    "get_enum_types",
    "parse_enum_table",
    "extract",
]
