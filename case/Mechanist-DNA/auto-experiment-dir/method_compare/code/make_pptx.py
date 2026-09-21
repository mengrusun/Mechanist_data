"""Place the exported figure on a 16:9 PPTX slide as a vector SVG picture.

PowerPoint renders the SVG (right-click > Convert to Shape makes it editable);
older viewers fall back to the 600-dpi PNG that is embedded alongside it.
"""
import os, sys, copy
sys.path.insert(0, os.path.dirname(__file__))
import mc_env
from pptx import Presentation
from pptx.util import Inches, Emu
from pptx.oxml.ns import qn
from lxml import etree

FIG = os.path.join(mc_env.MC, "figures")
STEM = os.path.join(FIG, "R1_compute_scaling")
SVG_URI = "{96DAC541-7B7A-43D3-8B79-37D633B846F1}"
NS_ASVG = "http://schemas.microsoft.com/office/drawing/2016/SVG/main"
SVG_CT = "image/svg+xml"

FIG_W_IN, FIG_H_IN = 7.09, 3.27          # the figure's true size, 180 mm wide
SLIDE_W_IN, SLIDE_H_IN = 13.333, 7.5
PIC_W_IN = 11.0                           # on-slide width


def add_svg_picture(slide, svg_path, png_path, left, top, width):
    """Add the PNG, then attach the SVG as the vector source of the same picture."""
    pic = slide.shapes.add_picture(png_path, left, top, width=width)

    part = slide.part
    svg_part, svg_rId = part.get_or_add_image_part(png_path)   # placeholder, replaced below
    # register the SVG as its own image part with the right content type
    image_part = part.package.get_or_add_image_part(svg_path)
    image_part._blob_content_type = SVG_CT
    svg_rId = part.relate_to(image_part, pic._element.blip_rId and
                             "http://schemas.openxmlformats.org/officeDocument/"
                             "2006/relationships/image")

    blip = pic._element.blipFill.find(qn("a:blip"))
    ext_lst = etree.SubElement(blip, qn("a:extLst"))
    ext = etree.SubElement(ext_lst, qn("a:ext"))
    ext.set("uri", SVG_URI)
    svg_blip = etree.SubElement(ext, "{%s}svgBlip" % NS_ASVG)
    svg_blip.set(qn("r:embed"), svg_rId)
    return pic


def main():
    prs = Presentation()
    prs.slide_width = Inches(SLIDE_W_IN)
    prs.slide_height = Inches(SLIDE_H_IN)
    slide = prs.slides.add_slide(prs.slide_layouts[6])      # blank

    pic_h_in = PIC_W_IN * FIG_H_IN / FIG_W_IN
    left = Emu(int((SLIDE_W_IN - PIC_W_IN) / 2 * 914400))
    top = Emu(int((SLIDE_H_IN - pic_h_in) / 2 * 914400))
    add_svg_picture(slide, STEM + ".svg", STEM + ".png", left, top,
                    Inches(PIC_W_IN))

    out = STEM + ".pptx"
    prs.save(out)
    print("WROTE", out)


if __name__ == "__main__":
    main()
