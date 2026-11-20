-- 1. Páginas por idioma
SELECT idioma, count(*) AS paginas
FROM plata_documentos
GROUP BY idioma
ORDER BY paginas DESC;

-- 2. Páginas por sección del sitio (primera parte de la URL)
SELECT split_part(url, '/', 4) AS seccion, count(*) AS paginas
FROM plata_documentos
GROUP BY 1
ORDER BY paginas DESC
LIMIT 40;