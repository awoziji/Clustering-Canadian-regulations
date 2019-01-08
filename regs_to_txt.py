import os
from multiprocessing import cpu_count
from multiprocessing.pool import Pool
from typing import List

from lxml import etree


def xml_to_str(filename: str) -> str:
    '''
    Extract text from an xml file.
    TODO: Some text is redundant and might affect the result of clustering. Better to filter out
    :param filename: xml filename
    :type filename: str
    :return: text extracted from xml file
    :rtype: str
    '''
    tree = etree.parse(filename)
    text_elems: List = tree.findall('.//Text')
    return '\n'.join(map(lambda el: el.xpath('string()'), text_elems))


def xml_to_txt(xml: str) -> None:
    """
    Extract text from an xml file and save in txt w/ same name.
    Used for multiprocessing.
    :param xml: xml filename
    :type xml: str
    """
    target_file: str = os.path.join(target_dir, xml.split('.xml')[0])
    with open(target_file, 'w') as f:
        f.write(xml_to_str(xml))


if __name__ == '__main__':
    # Create dir to store txt's
    target_dir = os.path.join(os.getcwd(), 'txt')
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)

    # List all xml's for regulations
    regs_dir: str = os.path.join(os.getcwd(), '../data/data/Consolidation_Regs_1.2.0/EN')
    os.chdir(regs_dir)
    xmls: List[str] = [filename for filename in os.listdir(regs_dir) if
                       filename.endswith('.xml')]

    # Convert xml to txt in parallel
    with Pool(cpu_count() - 2) as p:
        p.map(xml_to_txt, xmls)
