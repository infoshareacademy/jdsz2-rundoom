-- Ilość wniosków wg dni tygodnia

SELECT date_part('dow', data_utworzenia) AS dzien_tygodnia, COUNT(1)
FROM wnioski
GROUP BY dzien_tygodnia;

-- Ilość wniosków wg dni tygodnia z podziałem na lata

SELECT date_part('dow', data_utworzenia) AS dzien_tygodnia,
       COUNT(CASE WHEN date_part('year', data_utworzenia) = 2013 THEN id END) AS rok_2013,
       COUNT(CASE WHEN date_part('year', data_utworzenia) = 2014 THEN id END) AS rok_2014,
       COUNT(CASE WHEN date_part('year', data_utworzenia) = 2015 THEN id END) AS rok_2015,
       COUNT(CASE WHEN date_part('year', data_utworzenia) = 2016 THEN id END) AS rok_2016,
       COUNT(CASE WHEN date_part('year', data_utworzenia) = 2017 THEN id END) AS rok_2017,
       COUNT(1) AS razem
FROM wnioski
GROUP BY dzien_tygodnia;

-- Ilość wniosków wg dni miesiąca

SELECT date_part('month', data_utworzenia) AS miesiac, COUNT(1)
FROM wnioski
GROUP BY miesiac;

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

-- Ilość wniosków wg kwartałów

SELECT date_part('quarter', data_utworzenia) AS kwartal, COUNT(1)
FROM wnioski
GROUP BY kwartal;

-- Ilość wniosków wg kwartałów z podziałem na lata

SELECT date_part('quarter', data_utworzenia) AS kwartal,
       COUNT(CASE WHEN date_part('year', data_utworzenia) = 2013 THEN id END) AS rok_2013,
       COUNT(CASE WHEN date_part('year', data_utworzenia) = 2014 THEN id END) AS rok_2014,
       COUNT(CASE WHEN date_part('year', data_utworzenia) = 2015 THEN id END) AS rok_2015,
       COUNT(CASE WHEN date_part('year', data_utworzenia) = 2016 THEN id END) AS rok_2016,
       COUNT(CASE WHEN date_part('year', data_utworzenia) = 2017 THEN id END) AS rok_2017,
       COUNT(1) AS razem
FROM wnioski
GROUP BY kwartal;