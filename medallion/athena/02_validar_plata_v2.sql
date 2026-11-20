-- 1. Versiones (snapshots) guardadas de la tabla plata
SELECT snapshot_id, committed_at
FROM "plata_documentos$snapshots"
ORDER BY committed_at;

-- 2. Líneas de solo números o códigos que había en la versión 1
--    (snapshot 578503986363922103 = plata v1, 2026-11-19 23:07 UTC)
SELECT linea, count(DISTINCT pagina_id) AS paginas, count(*) AS veces
FROM plata_documentos FOR VERSION AS OF 578503986363922103
CROSS JOIN UNNEST(split(texto, chr(10))) AS t(linea)
WHERE regexp_like(linea, '^(\d+|#[0-9a-fA-F]{3,8})$')
GROUP BY linea
ORDER BY veces DESC
LIMIT 50;

-- 3. Líneas que más se repiten en la versión 2
SELECT linea, count(DISTINCT pagina_id) AS paginas
FROM plata_documentos
CROSS JOIN UNNEST(split(texto, chr(10))) AS t(linea)
GROUP BY linea
ORDER BY paginas DESC
LIMIT 30;

-- 4. Rechazadas en la versión 2
SELECT motivo_rechazo, url, titulo
FROM plata_rechazos
ORDER BY motivo_rechazo, url;

-- 5. Texto de tarifas-generales en la versión 1 (antes de que R4 la quitara)
SELECT n_caracteres, texto
FROM plata_documentos FOR VERSION AS OF 578503986363922103
WHERE url = 'https://www.eafit.edu.co/tarifas-generales';