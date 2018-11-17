-- Ilość wniosków wg kwartałów z podziałem na lata
-- Rok 2013 odrzucono, ze względu na małą istotność
-- Rok 2018 odrzucono, ze względu na to, że zawierał dane tylko z 2 pierwszych miesiecy

SELECT date_part('quarter', data_utworzenia) AS kwartal,
       COUNT(CASE WHEN date_part('year', data_utworzenia) = 2014 THEN id END) AS rok_2014,
       COUNT(CASE WHEN date_part('year', data_utworzenia) = 2015 THEN id END) AS rok_2015,
       COUNT(CASE WHEN date_part('year', data_utworzenia) = 2016 THEN id END) AS rok_2016,
       COUNT(CASE WHEN date_part('year', data_utworzenia) = 2017 THEN id END) AS rok_2017
FROM wnioski
GROUP BY kwartal;

-- Kwartał słownie

SELECT to_char(data_utworzenia, 'quarter') AS kwartal,
       COUNT(CASE WHEN date_part('year', data_utworzenia) = 2014 THEN id END) AS rok_2014,
       COUNT(CASE WHEN date_part('year', data_utworzenia) = 2015 THEN id END) AS rok_2015,
       COUNT(CASE WHEN date_part('year', data_utworzenia) = 2016 THEN id END) AS rok_2016,
       COUNT(CASE WHEN date_part('year', data_utworzenia) = 2017 THEN id END) AS rok_2017
FROM wnioski
GROUP BY kwartal;

-- Bardziej rozbudowana analityka

WITH wnioski_quarter as (
SELECT date_part('quarter', data_utworzenia) AS quarter,
       COUNT(CASE WHEN date_part('year', data_utworzenia) = 2014 THEN id END) AS rok_2014,
       COUNT(CASE WHEN date_part('year', data_utworzenia) = 2015 THEN id END) AS rok_2015,
       COUNT(CASE WHEN date_part('year', data_utworzenia) = 2016 THEN id END) AS rok_2016,
       COUNT(CASE WHEN date_part('year', data_utworzenia) = 2017 THEN id END) AS rok_2017,
       COUNT(CASE WHEN date_part('year', data_utworzenia) BETWEEN 2014 AND 2017 THEN id END) AS razem
FROM wnioski
GROUP BY quarter)

SELECT *,
round(avg(razem) over(), 0) as srednia_razem,
round(stddev_pop(razem) over(), 0) as odch_std
FROM wnioski_quarter;