-- Ilość wniosków wg dni miesiąca z podziałem na lata

SELECT date_part('month', data_utworzenia) AS miesiac,
       COUNT(CASE WHEN date_part('year', data_utworzenia) = 2013 THEN id END) AS rok_2013,
       COUNT(CASE WHEN date_part('year', data_utworzenia) = 2014 THEN id END) AS rok_2014,
       COUNT(CASE WHEN date_part('year', data_utworzenia) = 2015 THEN id END) AS rok_2015,
       COUNT(CASE WHEN date_part('year', data_utworzenia) = 2016 THEN id END) AS rok_2016,
       COUNT(CASE WHEN date_part('year', data_utworzenia) = 2017 THEN id END) AS rok_2017,
       COUNT(1) AS razem
FROM wnioski
GROUP BY miesiac;