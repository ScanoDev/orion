-- 1. Tamaño de las páginas limpias
SELECT count(*) AS paginas,
       round(avg(n_caracteres)) AS promedio,
       approx_percentile(n_caracteres, 0.5) AS mediana,
       min(n_caracteres) AS minimo,
       max(n_caracteres) AS maximo
FROM plata_documentos;

-- 2. Líneas que todavía se repiten en muchas páginas (menús que se escaparon)
SELECT linea, count(DISTINCT pagina_id) AS paginas
FROM plata_documentos
CROSS JOIN UNNEST(split(texto, chr(10))) AS t(linea)
GROUP BY linea
ORDER BY paginas DESC
LIMIT 30;

-- 3. Una página real, para leerla a ojo
SELECT url, titulo, substr(texto, 1, 1500) AS inicio
FROM plata_documentos
WHERE url LIKE '%ingenieria-de-sistemas%'
LIMIT 3;

-- 4. Las páginas rechazadas y su motivo
SELECT motivo_rechazo, url, titulo
FROM plata_rechazos
ORDER BY motivo_rechazo, url;

-- 6. Líneas repetidas en rango medio (para calibrar el umbral de R2)
SELECT linea, count(DISTINCT pagina_id) AS paginas
FROM plata_documentos
CROSS JOIN UNNEST(split(texto, chr(10))) AS t(linea)
GROUP BY linea
HAVING count(DISTINCT pagina_id) BETWEEN 100 AND 371
ORDER BY paginas DESC
LIMIT 50;

-- 7. Líneas repetidas entre 100 y 229 páginas (para fijar el umbral)
SELECT linea, count(DISTINCT pagina_id) AS paginas
FROM plata_documentos
CROSS JOIN UNNEST(split(texto, chr(10))) AS t(linea)
GROUP BY linea
HAVING count(DISTINCT pagina_id) BETWEEN 100 AND 229
ORDER BY paginas DESC;