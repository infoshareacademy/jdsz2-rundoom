
select count(1),jezyk from wnioski
group by jezyk

select count(1), typ_wniosku from wnioski
group by typ_wniosku

select  count(1), zrodlo_polecenia from wnioski
group by zrodlo_polecenia
