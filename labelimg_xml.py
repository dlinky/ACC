import os
import xml.etree.ElementTree as ET


def read_xml(path, filename):
    print('reading', filename)
    # 파일 불러오기
    doc = ET.parse(path + filename)
    # 최상단 태그 지정 : 'annotation'
    root = doc.getroot()

    # 레이블만 빼놓은것. 아마 안쓸듯
    labels_title = ['folder', 'filename', 'path', 'database', 'width', 'height', 'depth', 'segmented']

    title = []  # 파일, 이미지 정보 등 저장
    table = []  # bounding box 저장

    # 파일정보, 이미지정보 읽어와서 title에 저장
    for child in list(root):
        # 2단계 자식노드 (source, size) 따로 처리
        if list(child):
            # bndbox 따로 처리
            if child.tag == 'object':
                continue
            else:
                for grandchild in list(child):
                    title.append(grandchild.text)
        # 1단계 자식노드
        else:
            title.append(child.text)

    # bounding box 저장
    for object in root.iter('object'):
        line = [object.findtext('name')]
        line.extend([int(point.text) for point in list(object.find('bndbox'))])
        table.append(line.copy())
        line.clear()
    return title, table


def write_xml(title, table, path, filename):
    print('writing xml', end=' ', flush=True)
    root = ET.Element('annotation')
    ET.SubElement(root, 'folder').text = title[0]
    ET.SubElement(root, 'filename').text = title[1]
    ET.SubElement(root, 'path').text = title[2]

    source = ET.SubElement(root, 'source')
    ET.SubElement(source, 'database').text = title[3]

    size = ET.SubElement(root, 'size')
    ET.SubElement(size, 'width').text = title[4]
    ET.SubElement(size, 'height').text = title[5]
    ET.SubElement(size, 'depth').text = title[6]

    ET.SubElement(root, 'segmented').text = title[7]

    for line in table:
        obj = ET.SubElement(root, 'object')
        ET.SubElement(obj, 'name').text = line[0]
        ET.SubElement(obj, 'pose').text = 'Unspecified'
        ET.SubElement(obj, 'truncated').text = '0'
        ET.SubElement(obj, 'difficult').text = '0'

        bndbox = ET.SubElement(obj, 'bndbox')
        ET.SubElement(bndbox, 'xmin').text = str(line[1])
        ET.SubElement(bndbox, 'ymin').text = str(line[2])
        ET.SubElement(bndbox, 'xmax').text = str(line[3])
        ET.SubElement(bndbox, 'ymax').text = str(line[4])

    tree = ET.ElementTree(root)
    tree.write(path + filename)
    print(', ', end='', flush=True)


def main():
    title, table = read_xml(os.getcwd(), '/sample.xml')
    write_xml(title, table, os.getcwd(), '/result.xml')


if __name__ == '__main__':
    main()
