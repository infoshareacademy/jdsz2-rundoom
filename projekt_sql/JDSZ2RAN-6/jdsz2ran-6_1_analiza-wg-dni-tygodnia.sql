-- Ilość wniosków wg dni tygodnia z podziałem na lata
-- Rok 2013 odrzucono, ze względu na małą istotność
-- Rok 2018 odrzucono, ze względu na, że zawierał dane tylko z 2 pierwszych miesiecy

SELECT date_part('dow', data_utworzenia) AS dzien_tygodnia,
       COUNT(CASE WHEN date_part('year', data_utworzenia) = 2014 THEN id END) AS rok_2014,
       COUNT(CASE WHEN date_part('year', data_utworzenia) = 2015 THEN id END) AS rok_2015,
       COUNT(CASE WHEN date_part('year', data_utworzenia) = 2016 THEN id END) AS rok_2016,
       COUNT(CASE WHEN date_part('year', data_utworzenia) = 2017 THEN id END) AS rok_2017
FROM wnioski
GROUP BY dzien_tygodnia;

-- Dni tygodnia słownie

SELECT to_char(data_utworzenia, 'day') AS dzien_tygodnia,
       COUNT(CASE WHEN date_part('year', data_utworzenia) = 2014 THEN id END) AS rok_2014,
       COUNT(CASE WHEN date_part('year', data_utworzenia) = 2015 THEN id END) AS rok_2015,
       COUNT(CASE WHEN date_part('year', data_utworzenia) = 2016 THEN id END) AS rok_2016,
       COUNT(CASE WHEN date_part('year', data_utworzenia) = 2017 THEN id END) AS rok_2017
FROM wnioski
GROUP BY dzien_tygodnia;

-- Bardziej rozbudowana analityka

WITH wnioski_dow as (
SELECT date_part('dow', data_utworzenia) AS dzien_tygodnia,
       COUNT(CASE WHEN date_part('year', data_utworzenia) = 2014 THEN id END) AS rok_2014,
       COUNT(CASE WHEN date_part('year', data_utworzenia) = 2015 THEN id END) AS rok_2015,
       COUNT(CASE WHEN date_part('year', data_utworzenia) = 2016 THEN id END) AS rok_2016,
       COUNT(CASE WHEN date_part('year', data_utworzenia) = 2017 THEN id END) AS rok_2017,
       COUNT(CASE WHEN date_part('year', data_utworzenia) BETWEEN 2014 AND 2017 THEN id END) AS razem
FROM wnioski
GROUP BY dzien_tygodnia)

SELECT *,
round(avg(razem) over(), 0) as srednia_razem,
round(stddev_pop(razem) over(), 0) as odch_std
FROM wnioski_dow;