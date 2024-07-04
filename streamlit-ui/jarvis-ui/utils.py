import hashlib
import os
from pdf2image import convert_from_path
from PIL import ImageDraw
import fitz

def generate_hasher(unique_id):
    # Create a new sha256 hash object
    sha_signature = hashlib.sha256(unique_id.encode()).hexdigest()
    return sha_signature

def generate_tenant_id(username, password):
    return generate_hasher(username + password)

def draw_bounding_box_on_pdf_image(doc, dpi=200):
    pdf_path = doc["file_path"]
    page_number = doc["page_idx"]
    coordinates = doc["bbox"]

    # Convert PDF page to image
    images = convert_from_path(pdf_path, first_page=page_number + 1, last_page=page_number + 1, dpi=dpi)
    
    # Assuming we have only one image since we specified a single page
    img = images[0]
    
    # Get the size of the PDF page in points (1 point = 1/72 inches)
    pdf_doc = fitz.open(pdf_path)
    page = pdf_doc.load_page(page_number)
    page_width, page_height = page.rect.width, page.rect.height
    
    # Scale coordinates to match the image resolution
    scale_x = img.width / page_width
    scale_y = img.height / page_height
    scaled_coordinates = tuple(int(coord * max(scale_x, scale_y)) for coord in coordinates)

    # Manually tune the right limit more
    scaled_coordinates = (scaled_coordinates[0], scaled_coordinates[1], scaled_coordinates[2] + 60, scaled_coordinates[3])
    
    # Draw the bounding box on the image
    draw = ImageDraw.Draw(img)
    draw.rectangle(scaled_coordinates, outline="red", width=2)

    # Image file name
    # take the pdf_path and add page_number
    # image_path = pdf_path.replace(".pdf", f"_page_{page_number}_{coordinates[1]}.png")
    image_path = f"output/{pdf_path.split("/")[-1].replace(".pdf", f"_page_{page_number}_{coordinates[1]}.png")}"
    
    # Save the image with the bounding box
    img.save(image_path)

    return image_path

def delete_screenshots():
    # delete all the png files in output dir
    for file in os.listdir("output/"):
        if file.endswith(".png"):
            os.remove(f"output/{file}")