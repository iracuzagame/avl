import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
import time

def download_images_and_save_html(url, output_folder):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f'Carpeta creada: {output_folder}')


    print(f'Obteniendo contenido de la página: {url}')
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.get(url)


    categories = driver.find_elements(By.CSS_SELECTOR, 'li a[data-filter]')
    all_img_urls = []

    for category in categories:
        category.click()
        time.sleep(10)


        soup = BeautifulSoup(driver.page_source, 'html.parser')


        img_tags = soup.find_all('img')


        srcset_images = []
        for img_tag in img_tags:
            srcset = img_tag.get('srcset')
            if srcset:
                srcset_urls = [urljoin(url, img_url.split(' ')[0]) for img_url in srcset.split(',')]
                srcset_images.extend(srcset_urls)


        style_images = []
        for tag in soup.find_all(style=True):
            style = tag['style']
            if 'background-image' in style:
                start = style.find('url(') + 4
                end = style.find(')', start)
                img_url = style[start:end].strip('\'"')
                style_images.append(urljoin(url, img_url))


        img_urls = [urljoin(url, img_tag.get('src')) for img_tag in img_tags if img_tag.get('src')]
        img_urls.extend(srcset_images)
        img_urls.extend(style_images)
        all_img_urls.extend(img_urls)

    driver.quit()
    all_img_urls = list(set(all_img_urls))

    print(f'Total de imágenes encontradas: {len(all_img_urls)}')

    for img_url in all_img_urls:

        img_name = os.path.basename(img_url)

        img_path = os.path.join(output_folder, img_name)

        try:
            print(f'Descargando {img_url}...')
            img_data = requests.get(img_url).content
            with open(img_path, 'wb') as handler:
                handler.write(img_data)
            print(f'Imagen descargada: {img_path}')
        except Exception as e:
            print(f'Error al descargar la imagen {img_url}: {e}')


    html_output_path = os.path.join(output_folder, 'contenido.html')
    with open(html_output_path, 'w', encoding='utf-8') as html_file:
        html_file.write(soup.prettify())
    print(f'Contenido HTML guardado en: {html_output_path}')


url = 'https://365monitoreo.com/dispositivos'
output_folder = 'web'
download_images_and_save_html(url, output_folder)
